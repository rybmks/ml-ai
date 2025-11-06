use std::collections::VecDeque;
use std::time::Instant;

use super::SearchAlgorithm;
use super::successors;
use crate::types::*;

#[derive(Debug, Clone)]
pub struct Bfs;

impl SearchAlgorithm for Bfs {
    fn name(&self) -> &'static str {
        "BFS"
    }

    fn run(&self, instanse: &Instance, start_state: &State, limit: &Limits) -> Outcome {
        let mut queue = VecDeque::new();
        queue.push_back(start_state.clone());

        let mut generated = 1u64;
        let mut max_in_mem = 1u64;
        let start_time = Instant::now();

        while let Some(state) = queue.pop_front() {
            if start_time.elapsed() > limit.time {
                return Outcome::new_unsolved(generated, max_in_mem);
            }

            if state.is_goal() {
                return Outcome::new_solved(state, generated, max_in_mem);
            }

            for new_state in successors(&state, instanse) {
                generated += 1;
                queue.push_back(new_state);
            }

            max_in_mem = max_in_mem.max(queue.len() as u64);

            if queue.len() as u64 >= limit.mem as u64 {
                return Outcome::new_unsolved(generated, max_in_mem);
            }
        }

        Outcome::new_unsolved(generated, max_in_mem)
    }
}
