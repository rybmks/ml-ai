use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::time::Instant;

use super::SearchAlgorithm;
use super::successors;
use crate::heuristic::heuristic;
use crate::types::*;

#[derive(Clone, Eq, PartialEq)]
struct AStarNode {
    f_val: u32,
    h_val: u32,
    g_val: u32,
    state: State,
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_val
            .cmp(&self.f_val)
            .then_with(|| self.h_val.cmp(&other.h_val))
            .then_with(|| other.g_val.cmp(&self.g_val))
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct AStar;

impl SearchAlgorithm for AStar {
    fn name(&self) -> &'static str {
        "A*"
    }

    fn run(&self, instanse: &Instance, start_state: &State, limit: &Limits) -> Outcome {
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();

        let h_start = heuristic(start_state, instanse);
        let mut start_node = AStarNode {
            f_val: start_state.g_val + h_start,
            g_val: start_state.g_val,
            h_val: h_start,
            state: start_state.clone(),
        };
        start_node.state.h_val = h_start;
        start_node.state.f_val = start_state.g_val + h_start;

        open_set.push(start_node);

        let mut generated = 1u64;
        let mut max_in_mem = 1u64;
        let start_time = Instant::now();

        while let Some(AStarNode { state, .. }) = open_set.pop() {
            if start_time.elapsed() > limit.time {
                return Outcome::new_unsolved(generated, max_in_mem);
            }

            if state.is_goal() {
                return Outcome::new_solved(state, generated, max_in_mem);
            }

            if !closed_set.insert(state.clone()) {
                continue;
            }

            for mut new_state in successors(&state, instanse) {
                generated += 1;

                let h_new = heuristic(&new_state, instanse);
                new_state.h_val = h_new;
                new_state.f_val = new_state.g_val + h_new;

                let new_node = AStarNode {
                    f_val: new_state.f_val,
                    g_val: new_state.g_val,
                    h_val: new_state.h_val,
                    state: new_state,
                };

                open_set.push(new_node);
            }

            max_in_mem = max_in_mem.max(open_set.len() as u64 + closed_set.len() as u64);

            if max_in_mem >= limit.mem as u64 {
                return Outcome::new_unsolved(generated, max_in_mem);
            }
        }

        Outcome::new_unsolved(generated, max_in_mem)
    }
}
