/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use crate::data_type::DataType;
pub trait Array: std::fmt::Debug + Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn data_type(&self) -> DataType;
    fn as_any(&self) -> &dyn std::any::Any;
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub data_type: DataType,
    pub values: Arc<dyn Array>,
    pub validity_bits: Option<Vec<u64>>,
}

impl Column {
    pub fn new<T: Array + 'static>(name: &str, values: Arc<T>) -> Self {
        Self { name: name.to_string(), data_type: values.data_type(), values, validity_bits: None }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn have_null(&self) -> bool {
        self.validity_bits.is_some()
    }
}
