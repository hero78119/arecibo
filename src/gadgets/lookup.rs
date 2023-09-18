//! This module implements lookup gadget for applications built with Nova.
use std::cmp::max;
use std::collections::BTreeMap;

use bellpepper::gadgets::Assignment;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};

use crate::constants::NUM_CHALLENGE_BITS;
use crate::gadgets::nonnative::util::Num;
use crate::gadgets::utils::alloc_const;
use crate::spartan::math::Math;
use crate::traits::ROCircuitTrait;
use crate::traits::ROConstants;
use crate::traits::ROTrait;
use crate::traits::{Group, ROConstantsCircuit};
use ff::{Field, PrimeField};

use super::utils::scalar_as_base;
use super::utils::{alloc_one, conditionally_select2, le_bits_to_num};

/// rw trace
pub enum RWTrace<T> {
  /// read
  Read(T, T),
  /// write
  Write(T, T),
}

/// Lookup in R1CS
#[derive(Clone, Debug, PartialEq)]
pub enum TableType {
  /// read only
  ReadOnly,
  /// write
  ReadWrite,
}

/// for starting a transaction simulation
pub struct LookupTransactionSimulate<'a, G: Group> {
  lookup: &'a mut Lookup<G::Base>,
  rw_trace: Vec<RWTrace<G::Base>>,
  map_aux: BTreeMap<G::Base, (G::Base, G::Base)>,
}

impl<'a, G: Group> LookupTransactionSimulate<'a, G> {
  /// start a new transaction simulated
  pub fn start_transaction(lookup: &'a mut Lookup<G::Base>) -> LookupTransactionSimulate<'a, G> {
    LookupTransactionSimulate {
      lookup,
      rw_trace: vec![],
      map_aux: BTreeMap::new(),
    }
  }

  /// read value from table
  pub fn read(&mut self, addr: G::Base) -> G::Base
  where
    <G as Group>::Base: std::cmp::Ord,
  {
    let key = &addr;
    let (value, _) = self.map_aux.entry(*key).or_insert_with(|| {
      self
        .lookup
        .map_aux
        .get(key)
        .cloned()
        .unwrap_or_else(|| (G::Base::from(0), G::Base::from(0)))
    });
    self.rw_trace.push(RWTrace::Read(addr, *value)); // append read trace
    *value
  }
  /// write value to lookup table
  pub fn write(&mut self, addr: G::Base, value: G::Base)
  where
    <G as Group>::Base: std::cmp::Ord,
  {
    let _ = self.map_aux.insert(
      addr,
      (
        value,
        G::Base::ZERO, // zero counter doens't matter, real counter will be computed inside lookup table
      ),
    );
    self.rw_trace.push(RWTrace::Write(addr, value)); // append read trace
  }
  /// commit rw_trace to lookup
  pub fn commit(&mut self, ro_consts: ROConstants<G>, prev_intermediate_gamma: G::Base) -> G::Base
  where
    <G as Group>::Base: std::cmp::Ord,
  {
    let mut hasher = <G as Group>::RO::new(ro_consts, 1 + self.rw_trace.len() * 3);
    hasher.absorb(prev_intermediate_gamma);

    self.rw_trace.iter().for_each(|rwtrace| {
      let (addr, (read_value, read_counter)) = match rwtrace {
        RWTrace::Read(addr, value) => (addr, self.lookup.rw_operation(true, *addr, *value)),
        RWTrace::Write(addr, value) => (addr, self.lookup.rw_operation(false, *addr, *value)),
      };
      hasher.absorb(*addr);
      hasher.absorb(read_value);
      hasher.absorb(read_counter);
    });
    let hash_bits = hasher.squeeze(NUM_CHALLENGE_BITS);
    scalar_as_base::<G>(hash_bits)
  }
}

/// for starting a transaction
pub struct LookupTransaction<'a, G: Group> {
  lookup: &'a mut Lookup<G::Base>,
  rw_trace: Vec<RWTrace<AllocatedNum<G::Base>>>,
  map_aux: BTreeMap<G::Base, (G::Base, G::Base)>,
}

impl<'a, G: Group> LookupTransaction<'a, G> {
  /// start a new transaction
  pub fn start_transaction(lookup: &'a mut Lookup<G::Base>) -> LookupTransaction<'a, G> {
    LookupTransaction {
      lookup,
      rw_trace: vec![],
      map_aux: BTreeMap::new(),
    }
  }

  /// read value from table
  pub fn read<CS: ConstraintSystem<<G as Group>::Base>>(
    &mut self,
    mut cs: CS,
    addr: &AllocatedNum<G::Base>,
  ) -> Result<AllocatedNum<G::Base>, SynthesisError>
  where
    <G as Group>::Base: std::cmp::Ord,
  {
    let key = &addr.get_value().unwrap_or_default();
    let (value, _) = self.map_aux.entry(*key).or_insert_with(|| {
      self
        .lookup
        .map_aux
        .get(key)
        .cloned()
        .unwrap_or_else(|| (G::Base::from(0), G::Base::from(0)))
    });
    let read_value = AllocatedNum::alloc(cs.namespace(|| "read_value"), || Ok(*value))?;
    self.rw_trace.push(RWTrace::Read::<AllocatedNum<G::Base>>(
      addr.clone(),
      read_value.clone(),
    )); // append read trace
    Ok(read_value)
  }

  /// write value to lookup table
  pub fn write(
    &mut self,
    addr: &AllocatedNum<G::Base>,
    value: &AllocatedNum<G::Base>,
  ) -> Result<(), SynthesisError>
  where
    <G as Group>::Base: std::cmp::Ord,
  {
    let _ = self.map_aux.insert(
      addr.get_value().unwrap_or_default(),
      (
        value.get_value().unwrap_or_default(),
        G::Base::ZERO, // zero counter doens't matter, real counter will be computed inside lookup table
      ),
    );
    self
      .rw_trace
      .push(RWTrace::Write(addr.clone(), value.clone())); // append read trace
    Ok(())
  }

  /// commit rw_trace to lookup
  #[allow(clippy::too_many_arguments)]
  pub fn commit<CS: ConstraintSystem<<G as Group>::Base>>(
    &mut self,
    mut cs: CS,
    ro_const: ROConstantsCircuit<G>,
    prev_intermediate_gamma: &AllocatedNum<G::Base>,
    gamma: &AllocatedNum<G::Base>,
    prev_R: &AllocatedNum<G::Base>,
    prev_W: &AllocatedNum<G::Base>,
    prev_rw_counter: &AllocatedNum<G::Base>,
  ) -> Result<
    (
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
    ),
    SynthesisError,
  >
  where
    <G as Group>::Base: std::cmp::Ord,
  {
    let mut ro = G::ROCircuit::new(
      ro_const,
      1 + 3 * self.rw_trace.len(), // prev_challenge + [(address, value, counter)]
    );
    ro.absorb(prev_intermediate_gamma);
    let (next_R, next_W, next_rw_counter) = self.rw_trace.iter().enumerate().try_fold(
      (prev_R.clone(), prev_W.clone(), prev_rw_counter.clone()),
      |(prev_R, prev_W, prev_rw_counter), (i, rwtrace)| match rwtrace {
        RWTrace::Read(addr, read_value) => {
          let (next_R, next_W, next_rw_counter, read_value, read_counter) =
            self.lookup.rw_operation_circuit(
              cs.namespace(|| format!("{}th read ", i)),
              true,
              addr,
              gamma,
              read_value,
              &prev_R,
              &prev_W,
              &prev_rw_counter,
            )?;
          ro.absorb(addr);
          ro.absorb(&read_value);
          ro.absorb(&read_counter);
          Ok::<
            (
              AllocatedNum<G::Base>,
              AllocatedNum<G::Base>,
              AllocatedNum<G::Base>,
            ),
            SynthesisError,
          >((next_R, next_W, next_rw_counter))
        }
        RWTrace::Write(addr, write_value) => {
          let (next_R, next_W, next_rw_counter, read_value, read_counter) =
            self.lookup.rw_operation_circuit(
              cs.namespace(|| format!("{}th write ", i)),
              false,
              addr,
              gamma,
              write_value,
              &prev_R,
              &prev_W,
              &prev_rw_counter,
            )?;
          ro.absorb(addr);
          ro.absorb(&read_value);
          ro.absorb(&read_counter);
          Ok::<
            (
              AllocatedNum<G::Base>,
              AllocatedNum<G::Base>,
              AllocatedNum<G::Base>,
            ),
            SynthesisError,
          >((next_R, next_W, next_rw_counter))
        }
      },
    )?;
    let hash_bits = ro.squeeze(cs.namespace(|| "challenge"), NUM_CHALLENGE_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), &hash_bits)?;
    Ok((next_R, next_W, next_rw_counter, hash))
  }
}

/// Lookup in R1CS
#[derive(Clone, Debug)]
pub struct Lookup<F: PrimeField> {
  pub(crate) map_aux: BTreeMap<F, (F, F)>, // (value, counter)
  /// map_aux_dirty only include the modified fields of `map_aux`, thats why called dirty
  map_aux_dirty: BTreeMap<F, (F, F)>, // (value, counter)
  rw_counter: F,
  table_type: TableType,         // read only or read-write
  max_cap_rwcounter_log2: usize, // max cap for rw_counter operation in bits
}

impl<F: PrimeField> Lookup<F> {
  /// new lookup table
  pub fn new(
    max_cap_rwcounter: usize,
    table_type: TableType,
    initial_table: Vec<(F, F)>,
  ) -> Lookup<F>
  where
    F: std::cmp::Ord,
  {
    let max_cap_rwcounter_log2 = max_cap_rwcounter.log_2();
    Self {
      map_aux: initial_table
        .into_iter()
        .map(|(addr, value)| (addr, (value, F::ZERO)))
        .collect(),
      map_aux_dirty: BTreeMap::new(),
      rw_counter: F::ZERO,
      table_type,
      max_cap_rwcounter_log2,
    }
  }

  fn rw_operation(&mut self, is_read: bool, addr: F, external_value: F) -> (F, F)
  where
    F: std::cmp::Ord,
  {
    // write operations
    if !is_read {
      debug_assert!(self.table_type == TableType::ReadWrite) // table need to set as rw
    }
    let (_read_value, _read_counter) = self
      .map_aux
      .get(&addr)
      .cloned()
      .unwrap_or((F::from(0), F::from(0)));

    let (write_value, write_counter) = (
      if is_read { _read_value } else { external_value },
      if self.table_type == TableType::ReadOnly {
        _read_counter
      } else {
        max(self.rw_counter, _read_counter)
      } + F::ONE,
    );
    self.map_aux.insert(addr, (write_value, write_counter));
    self
      .map_aux_dirty
      .insert(addr, (write_value, write_counter));
    self.rw_counter = write_counter;
    (_read_value, _read_counter)
  }

  #[allow(clippy::too_many_arguments)]
  fn rw_operation_circuit<CS: ConstraintSystem<F>>(
    &mut self,
    mut cs: CS,
    is_read: bool,
    addr: &AllocatedNum<F>,
    // challenges: &(AllocatedNum<G::Base>, AllocatedNum<G::Base>),
    gamma: &AllocatedNum<F>,
    external_value: &AllocatedNum<F>,
    prev_R: &AllocatedNum<F>,
    prev_W: &AllocatedNum<F>,
    prev_rw_counter: &AllocatedNum<F>,
  ) -> Result<
    (
      AllocatedNum<F>,
      AllocatedNum<F>,
      AllocatedNum<F>,
      AllocatedNum<F>,
      AllocatedNum<F>,
    ),
    SynthesisError,
  >
  where
    F: std::cmp::Ord,
  {
    // extract challenge
    // get content from map
    // value are provided beforehand from outside, therefore here just constraints it
    let (_read_value, _read_counter) = self
      .map_aux
      .get(&addr.get_value().unwrap_or_default())
      .cloned()
      .unwrap_or((F::from(0), F::from(0)));

    let read_counter = AllocatedNum::alloc(cs.namespace(|| "counter"), || Ok(_read_counter))?;

    // external_read_value should match with _read_value
    if is_read {
      if let Some(external_read_value) = external_value.get_value() {
        assert_eq!(external_read_value, _read_value)
      }
    };

    // external_read_value should match with rw_counter witness
    if let Some(external_rw_counter) = prev_rw_counter.get_value() {
      assert_eq!(external_rw_counter, self.rw_counter)
    }

    let one = F::ONE;

    // update R
    let gamma_square = gamma.mul(cs.namespace(|| "gamme^2"), gamma)?;
    // read_value_term = gamma * value
    let read_value = if is_read {
      external_value.clone()
    } else {
      AllocatedNum::alloc(cs.namespace(|| "read_value"), || Ok(_read_value))?
    };
    let read_value_term = gamma.mul(cs.namespace(|| "read_value_term"), &read_value)?;
    // counter_term = gamma^2 * counter
    let read_counter_term =
      gamma_square.mul(cs.namespace(|| "read_counter_term"), &read_counter)?;
    // new_R = R * (gamma - (addr + gamma * value + gamma^2 * counter))
    let new_R = AllocatedNum::alloc(cs.namespace(|| "new_R"), || {
      prev_R
        .get_value()
        .zip(gamma.get_value())
        .zip(addr.get_value())
        .zip(read_value_term.get_value())
        .zip(read_counter_term.get_value())
        .map(|((((R, gamma), addr), value_term), counter_term)| {
          R * (gamma - (addr + value_term + counter_term))
        })
        .ok_or(SynthesisError::AssignmentMissing)
    })?;
    let mut r_blc = LinearCombination::<F>::zero();
    r_blc = r_blc + gamma.get_variable()
      - addr.get_variable()
      - read_value_term.get_variable()
      - read_counter_term.get_variable();
    cs.enforce(
      || "R update",
      |lc| lc + prev_R.get_variable(),
      |_| r_blc,
      |lc| lc + new_R.get_variable(),
    );

    // RO to get challenge
    // where the input only cover
    // ro.absorb(addr);
    // ro.absorb(&read_value);
    // ro.absorb(&read_counter);

    let alloc_num_one = alloc_one(cs.namespace(|| "one"))?;

    // max{read_counter, rw_counter} logic on read-write lookup
    // read_counter on read-only
    // - max{read_counter, rw_counter} if read-write table
    // - read_counter if read-only table
    // +1 will be hadle later
    let (write_counter, write_counter_term) = if self.table_type == TableType::ReadWrite {
      // write_counter = read_counter < prev_rw_counter ? prev_rw_counter: read_counter
      // TODO optimise with `max` table lookup to save more constraints
      let lt = less_than(
        cs.namespace(|| "read_counter < a"),
        &read_counter,
        prev_rw_counter,
        self.max_cap_rwcounter_log2,
      )?;
      let write_counter = conditionally_select2(
        cs.namespace(|| {
          "write_counter = read_counter < prev_rw_counter ? prev_rw_counter: read_counter"
        }),
        prev_rw_counter,
        &read_counter,
        &lt,
      )?;
      let write_counter_term =
        gamma_square.mul(cs.namespace(|| "write_counter_term"), &write_counter)?;
      (write_counter, write_counter_term)
    } else {
      (read_counter.clone(), read_counter_term)
    };

    // update W
    // write_value_term = gamma * value
    let write_value_term = if is_read {
      read_value_term
    } else {
      gamma.mul(cs.namespace(|| "write_value_term"), external_value)?
    };
    let new_W = AllocatedNum::alloc(cs.namespace(|| "new_W"), || {
      prev_W
        .get_value()
        .zip(gamma.get_value())
        .zip(addr.get_value())
        .zip(write_value_term.get_value())
        .zip(write_counter_term.get_value())
        .zip(gamma_square.get_value())
        .map(
          |(((((W, gamma), addr), value_term), write_counter_term), gamma_square)| {
            W * (gamma - (addr + value_term + write_counter_term + gamma_square))
          },
        )
        .ok_or(SynthesisError::AssignmentMissing)
    })?;
    // new_W = W * (gamma - (addr + gamma * value + gamma^2 * counter + gamma^2)))
    let mut w_blc = LinearCombination::<F>::zero();
    w_blc = w_blc + gamma.get_variable()
      - addr.get_variable()
      - write_value_term.get_variable()
      - write_counter_term.get_variable()
      - gamma_square.get_variable();
    cs.enforce(
      || "W update",
      |lc| lc + prev_W.get_variable(),
      |_| w_blc,
      |lc| lc + new_W.get_variable(),
    );

    // update witness
    self.map_aux.insert(
      addr.get_value().unwrap_or_default(),
      (
        external_value.get_value().unwrap_or_default(),
        write_counter.get_value().unwrap_or_default() + one,
      ),
    );
    self.map_aux_dirty.insert(
      addr.get_value().unwrap_or_default(),
      (
        external_value.get_value().unwrap_or_default(),
        write_counter.get_value().unwrap_or_default() + one,
      ),
    );
    let new_rw_counter = add_allocated_num(
      cs.namespace(|| "new_rw_counter"),
      &write_counter,
      &alloc_num_one,
    )?;
    if let Some(new_rw_counter) = new_rw_counter.get_value() {
      self.rw_counter = new_rw_counter;
    }
    Ok((new_R, new_W, new_rw_counter, read_value, read_counter))
  }

  // fn write(&mut self, addr: AllocatedNum<F>, value: F) {}
}

/// c = a + b where a, b is AllocatedNum
pub fn add_allocated_num<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "c"), || {
    Ok(*a.get_value().get()? + b.get_value().get()?)
  })?;
  cs.enforce(
    || "c = a+b",
    |lc| lc + a.get_variable() + b.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + c.get_variable(),
  );
  Ok(c)
}

/// a < b ? 1 : 0
pub fn less_than<F: PrimeField + PartialOrd, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  n_bits: usize,
) -> Result<AllocatedNum<F>, SynthesisError> {
  assert!(n_bits < 64, "not support n_bits {n_bits} >= 64");
  let range = alloc_const(
    cs.namespace(|| "range"),
    F::from(2_usize.pow(n_bits as u32) as u64),
  )?;
  // diff = (lhs - rhs) + (if lt { range } else { 0 });
  let diff = Num::alloc(cs.namespace(|| "diff"), || {
    a.get_value()
      .zip(b.get_value())
      .zip(range.get_value())
      .map(|((a, b), range)| {
        let lt = a < b;
        (a - b) + (if lt { range } else { F::ZERO })
      })
      .ok_or(SynthesisError::AssignmentMissing)
  })?;
  diff.fits_in_bits(cs.namespace(|| "diff fit in bits"), n_bits)?;
  let diff = diff.as_allocated_num(cs.namespace(|| "diff_alloc_num"))?;
  let lt = AllocatedNum::alloc(cs.namespace(|| "lt"), || {
    a.get_value()
      .zip(b.get_value())
      .map(|(a, b)| F::from((a < b) as u64))
      .ok_or(SynthesisError::AssignmentMissing)
  })?;
  cs.enforce(
    || "lt is bit",
    |lc| lc + lt.get_variable(),
    |lc| lc + CS::one() - lt.get_variable(),
    |lc| lc,
  );
  // println!(
  //   "a {:?}, b {:?}, diff {:?}, range {:?}, lt {:?}",
  //   a.get_value(),
  //   b.get_value(),
  //   diff.get_value(),
  //   range.get_value(),
  //   lt.get_value(),
  // );
  // println!(
  //   "{:?} =  {:?}",
  //   lt.get_value()
  //     .zip(range.get_value())
  //     .map(|(lt, range)| lt * range),
  //   diff
  //     .get_value()
  //     .zip(a.get_value())
  //     .zip(b.get_value())
  //     .map(|((diff, a), b)| diff - a + b),
  // );
  cs.enforce(
    || "lt ⋅ range == diff - lhs + rhs",
    |lc| lc + lt.get_variable(),
    |lc| lc + range.get_variable(),
    |lc| lc + diff.get_variable() - a.get_variable() + b.get_variable(),
  );
  Ok(lt)
}

#[cfg(test)]
mod test {
  use crate::{
    // bellpepper::test_shape_cs::TestShapeCS,
    constants::NUM_CHALLENGE_BITS,
    gadgets::{
      lookup::{LookupTransaction, LookupTransactionSimulate, TableType},
      utils::{alloc_one, alloc_zero, scalar_as_base},
    },
    provider::poseidon::PoseidonConstantsCircuit,
    traits::{Group, ROConstantsCircuit},
  };
  use ff::Field;

  use super::Lookup;
  use crate::traits::ROTrait;
  use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};

  #[test]
  fn test_lookup_simulation() {
    type G1 = pasta_curves::pallas::Point;
    type G2 = pasta_curves::vesta::Point;

    let ro_consts: ROConstantsCircuit<G2> = PoseidonConstantsCircuit::default();

    // let mut cs: TestShapeCS<G1> = TestShapeCS::new();
    let initial_table = vec![
      (<G1 as Group>::Scalar::ZERO, <G1 as Group>::Scalar::ZERO),
      (<G1 as Group>::Scalar::ONE, <G1 as Group>::Scalar::ONE),
    ];
    let mut lookup =
      Lookup::<<G1 as Group>::Scalar>::new(1024, TableType::ReadWrite, initial_table);
    let mut lookup_transaction = LookupTransactionSimulate::<G2>::start_transaction(&mut lookup);
    let prev_intermediate_gamma = <G1 as Group>::Scalar::ONE;
    let read_value = lookup_transaction.read(<G1 as Group>::Scalar::ZERO);
    assert_eq!(read_value, <G1 as Group>::Scalar::ZERO);
    let read_value = lookup_transaction.read(<G1 as Group>::Scalar::ONE);
    assert_eq!(read_value, <G1 as Group>::Scalar::ONE);
    lookup_transaction.write(
      <G1 as Group>::Scalar::ZERO,
      <G1 as Group>::Scalar::from(111),
    );
    let read_value = lookup_transaction.read(<G1 as Group>::Scalar::ZERO);
    assert_eq!(read_value, <G1 as Group>::Scalar::from(111),);

    let next_intermediate_gamma =
      lookup_transaction.commit(ro_consts.clone(), prev_intermediate_gamma);

    let mut hasher = <G2 as Group>::RO::new(ro_consts, 1 + 3 * 4);
    hasher.absorb(prev_intermediate_gamma);
    hasher.absorb(<G1 as Group>::Scalar::ZERO); // addr
    hasher.absorb(<G1 as Group>::Scalar::ZERO); // value
    hasher.absorb(<G1 as Group>::Scalar::ZERO); // counter
    hasher.absorb(<G1 as Group>::Scalar::ONE); // addr
    hasher.absorb(<G1 as Group>::Scalar::ONE); // value
    hasher.absorb(<G1 as Group>::Scalar::ZERO); // counter
    hasher.absorb(<G1 as Group>::Scalar::ZERO); // addr
    hasher.absorb(<G1 as Group>::Scalar::ZERO); // value
    hasher.absorb(<G1 as Group>::Scalar::ONE); // counter
    hasher.absorb(<G1 as Group>::Scalar::ZERO); // addr
    hasher.absorb(<G1 as Group>::Scalar::from(111)); // value
    hasher.absorb(<G1 as Group>::Scalar::from(3)); // counter
    let res = hasher.squeeze(NUM_CHALLENGE_BITS);
    assert_eq!(scalar_as_base::<G2>(res), next_intermediate_gamma);
  }

  #[test]
  fn test_read_twice_on_readonly() {
    type G1 = pasta_curves::pallas::Point;
    type G2 = pasta_curves::vesta::Point;

    let ro_consts: ROConstantsCircuit<G2> = PoseidonConstantsCircuit::default();

    let mut cs = TestConstraintSystem::<<G1 as Group>::Scalar>::new();
    // let mut cs: TestShapeCS<G1> = TestShapeCS::new();
    let initial_table = vec![
      (
        <G1 as Group>::Scalar::ZERO,
        <G1 as Group>::Scalar::from(101),
      ),
      (<G1 as Group>::Scalar::ONE, <G1 as Group>::Scalar::ZERO),
    ];
    let mut lookup = Lookup::<<G1 as Group>::Scalar>::new(1024, TableType::ReadOnly, initial_table);
    let mut lookup_transaction = LookupTransaction::<G2>::start_transaction(&mut lookup);
    let gamma = AllocatedNum::alloc(cs.namespace(|| "gamma"), || {
      Ok(<G1 as Group>::Scalar::from(2))
    })
    .unwrap();
    let zero = alloc_zero(cs.namespace(|| "zero")).unwrap();
    let one = alloc_one(cs.namespace(|| "one")).unwrap();
    let prev_intermediate_gamma = &one;
    let prev_rw_counter = &zero;
    let addr = zero.clone();
    let read_value = lookup_transaction
      .read(cs.namespace(|| "read_value1"), &addr)
      .unwrap();
    assert_eq!(
      read_value.get_value(),
      Some(<G1 as Group>::Scalar::from(101))
    );
    let read_value = lookup_transaction
      .read(cs.namespace(|| "read_value2"), &addr)
      .unwrap();
    assert_eq!(
      read_value.get_value(),
      Some(<G1 as Group>::Scalar::from(101))
    );
    let (prev_W, prev_R) = (&one, &one);
    let (next_R, next_W, next_rw_counter, next_intermediate_gamma) = lookup_transaction
      .commit(
        cs.namespace(|| "commit"),
        ro_consts.clone(),
        prev_intermediate_gamma,
        &gamma,
        prev_W,
        prev_R,
        prev_rw_counter,
      )
      .unwrap();
    assert_eq!(
      next_rw_counter.get_value(),
      Some(<G1 as Group>::Scalar::from(2))
    );
    // next_R check
    assert_eq!(
      next_R.get_value(),
      prev_R
        .get_value()
        .zip(gamma.get_value())
        .zip(addr.get_value())
        .zip(read_value.get_value())
        .map(|(((prev_R, gamma), addr), read_value)| prev_R
          * (gamma - (addr + gamma * read_value + gamma * gamma * <G1 as Group>::Scalar::ZERO))
          * (gamma - (addr + gamma * read_value + gamma * gamma * <G1 as Group>::Scalar::ONE)))
    );
    // next_W check
    assert_eq!(
      next_W.get_value(),
      prev_W
        .get_value()
        .zip(gamma.get_value())
        .zip(addr.get_value())
        .zip(read_value.get_value())
        .map(|(((prev_W, gamma), addr), read_value)| {
          prev_W
            * (gamma - (addr + gamma * read_value + gamma * gamma * (<G1 as Group>::Scalar::ONE)))
            * (gamma
              - (addr + gamma * read_value + gamma * gamma * (<G1 as Group>::Scalar::from(2))))
        }),
    );

    let mut hasher = <G2 as Group>::RO::new(ro_consts, 7);
    hasher.absorb(prev_intermediate_gamma.get_value().unwrap());
    hasher.absorb(addr.get_value().unwrap());
    hasher.absorb(read_value.get_value().unwrap());
    hasher.absorb(<G1 as Group>::Scalar::ZERO);
    hasher.absorb(addr.get_value().unwrap());
    hasher.absorb(read_value.get_value().unwrap());
    hasher.absorb(<G1 as Group>::Scalar::ONE);
    let res = hasher.squeeze(NUM_CHALLENGE_BITS);
    assert_eq!(
      scalar_as_base::<G2>(res),
      next_intermediate_gamma.get_value().unwrap()
    );
    // TODO check rics is_sat
    // let (_, _) = cs.r1cs_shape_with_commitmentkey();
    // let (U1, W1) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // // Make sure that the first instance is satisfiable
    // assert!(shape.is_sat(&ck, &U1, &W1).is_ok());
  }

  #[test]
  fn test_write_read_on_rwlookup() {
    type G1 = pasta_curves::pallas::Point;
    type G2 = pasta_curves::vesta::Point;

    let ro_consts: ROConstantsCircuit<G2> = PoseidonConstantsCircuit::default();

    let mut cs = TestConstraintSystem::<<G1 as Group>::Scalar>::new();
    // let mut cs: TestShapeCS<G1> = TestShapeCS::new();
    let initial_table = vec![
      (<G1 as Group>::Scalar::ZERO, <G1 as Group>::Scalar::ZERO),
      (<G1 as Group>::Scalar::ONE, <G1 as Group>::Scalar::ZERO),
    ];
    let mut lookup =
      Lookup::<<G1 as Group>::Scalar>::new(1024, TableType::ReadWrite, initial_table);
    let mut lookup_transaction = LookupTransaction::<G2>::start_transaction(&mut lookup);
    let gamma = AllocatedNum::alloc(cs.namespace(|| "gamma"), || {
      Ok(<G1 as Group>::Scalar::from(2))
    })
    .unwrap();
    let zero = alloc_zero(cs.namespace(|| "zero")).unwrap();
    let one = alloc_one(cs.namespace(|| "one")).unwrap();
    let prev_intermediate_gamma = &one;
    let prev_rw_counter = &zero;
    let addr = zero.clone();
    lookup_transaction
      .write(
        &addr,
        &AllocatedNum::alloc(cs.namespace(|| "write value 1"), || {
          Ok(<G1 as Group>::Scalar::from(101))
        })
        .unwrap(),
      )
      .unwrap();
    let read_value = lookup_transaction
      .read(cs.namespace(|| "read_value 1"), &addr)
      .unwrap();
    assert_eq!(
      read_value.get_value(),
      Some(<G1 as Group>::Scalar::from(101))
    );
    let (prev_W, prev_R) = (&one, &one);
    let (next_R, next_W, next_rw_counter, next_intermediate_gamma) = lookup_transaction
      .commit(
        cs.namespace(|| "commit"),
        ro_consts.clone(),
        prev_intermediate_gamma,
        &gamma,
        prev_W,
        prev_R,
        prev_rw_counter,
      )
      .unwrap();
    assert_eq!(
      next_rw_counter.get_value(),
      Some(<G1 as Group>::Scalar::from(2))
    );
    // next_R check
    assert_eq!(
      next_R.get_value(),
      prev_R
        .get_value()
        .zip(gamma.get_value())
        .zip(addr.get_value())
        .zip(read_value.get_value())
        .map(|(((prev_R, gamma), addr), read_value)| prev_R
          * (gamma
            - (addr
              + gamma * <G1 as Group>::Scalar::ZERO
              + gamma * gamma * <G1 as Group>::Scalar::ZERO))
          * (gamma - (addr + gamma * read_value + gamma * gamma * <G1 as Group>::Scalar::ONE)))
    );
    // next_W check
    assert_eq!(
      next_W.get_value(),
      prev_W
        .get_value()
        .zip(gamma.get_value())
        .zip(addr.get_value())
        .zip(read_value.get_value())
        .map(|(((prev_W, gamma), addr), read_value)| {
          prev_W
            * (gamma - (addr + gamma * read_value + gamma * gamma * (<G1 as Group>::Scalar::ONE)))
            * (gamma
              - (addr + gamma * read_value + gamma * gamma * (<G1 as Group>::Scalar::from(2))))
        }),
    );

    let mut hasher = <G2 as Group>::RO::new(ro_consts, 7);
    hasher.absorb(prev_intermediate_gamma.get_value().unwrap());
    hasher.absorb(addr.get_value().unwrap());
    hasher.absorb(<G1 as Group>::Scalar::ZERO);
    hasher.absorb(<G1 as Group>::Scalar::ZERO);
    hasher.absorb(addr.get_value().unwrap());
    hasher.absorb(read_value.get_value().unwrap());
    hasher.absorb(<G1 as Group>::Scalar::ONE);
    let res = hasher.squeeze(NUM_CHALLENGE_BITS);
    assert_eq!(
      scalar_as_base::<G2>(res),
      next_intermediate_gamma.get_value().unwrap()
    );
    // TODO check rics is_sat
    // let (_, _) = cs.r1cs_shape_with_commitmentkey();
    // let (U1, W1) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // // Make sure that the first instance is satisfiable
    // assert!(shape.is_sat(&ck, &U1, &W1).is_ok());
  }
}
