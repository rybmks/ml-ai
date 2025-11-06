pub mod astar;
pub mod bfs;

use bitvec::vec::BitVec;

use super::types::*;

pub trait SearchAlgorithm {
    fn name(&self) -> &'static str;
    fn run(&self, instanse: &Instance, start_state: &State, limit: &Limits) -> Outcome;
}

fn successors(state: &State, inst: &Instance) -> Vec<State> {
    let mut neighbors = Vec::new();

    for lesson_id in &state.remaining {
        let lesson = &inst.lessons[lesson_id.0];

        for r_idx in 0..inst.rooms.len() {
            let room = &inst.rooms[r_idx];

            let max_room_start = room.timeslots_count.saturating_sub(lesson.taken_timeslots);

            for start in 0..=max_room_start {
                if !fits_range(&state.room_busy[r_idx], start, lesson.taken_timeslots) {
                    continue;
                }

                for teacher_id in &lesson.candidate_teachers {
                    let t_idx = teacher_id.0;
                    let teacher = &inst.teachers[t_idx];
                    let teacher_busy_bv = &state.teacher_busy[t_idx];

                    if start + lesson.taken_timeslots > teacher.timeslots_count {
                        continue;
                    }

                    if fits_range(teacher_busy_bv, start, lesson.taken_timeslots) {
                        let mut new_state = state.clone();
                        new_state.g_val += 1;

                        set_range(
                            &mut new_state.room_busy[r_idx],
                            start,
                            lesson.taken_timeslots,
                            true,
                        );
                        set_range(
                            &mut new_state.teacher_busy[t_idx],
                            start,
                            lesson.taken_timeslots,
                            true,
                        );

                        new_state.placed.push(Placement {
                            lesson: lesson.id,
                            room: room.id,
                            teacher: *teacher_id,
                            start,
                            len: lesson.taken_timeslots,
                        });

                        new_state.remaining.retain(|&l| l != *lesson_id);
                        neighbors.push(new_state);
                    }
                }
            }
        }
    }

    neighbors
}

pub fn fits_range(bv: &BitVec, start: usize, len: usize) -> bool {
    let end = start + len;
    if end > bv.len() {
        return false;
    }

    bv[start..end].iter().all(|bit| !*bit)
}

pub fn set_range(bv: &mut BitVec, start: usize, len: usize, value: bool) {
    for i in start..start + len {
        bv.set(i, value);
    }
}
