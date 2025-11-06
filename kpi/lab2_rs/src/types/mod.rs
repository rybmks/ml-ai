use crate::search::SearchAlgorithm;
use bitvec::prelude::*;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LessonId(pub usize);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RoomId(pub usize);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TeacherId(pub usize);

#[derive(Clone, Debug)]
pub struct Room {
    pub id: RoomId,
    pub name: String,
    pub timeslots_count: usize,
}

impl Room {
    pub fn new<N: Into<String>>(id: usize, name: N, timeslots_count: usize) -> Self {
        Self {
            id: RoomId(id),
            name: name.into(),
            timeslots_count,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Teacher {
    pub id: TeacherId,
    pub name: String,
    pub timeslots_count: usize,
}

#[derive(Clone, Debug)]
pub struct Lesson {
    pub id: LessonId,
    pub name: String,
    pub taken_timeslots: usize,
    pub candidate_teachers: Vec<TeacherId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    pub room: RoomId,
    pub teacher: TeacherId,
    pub lesson: LessonId,
    pub start: usize,
    pub len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub room_busy: Vec<BitVec>,
    pub teacher_busy: Vec<BitVec>,
    pub remaining: Vec<LessonId>,
    pub placed: Vec<Placement>,

    pub g_val: u32,
    pub h_val: u32,
    pub f_val: u32,
}

impl State {
    pub fn new(inst: &Instance) -> Self {
        let room_busy = inst
            .rooms
            .iter()
            .map(|room| BitVec::repeat(false, room.timeslots_count))
            .collect();

        let teacher_busy = inst
            .teachers
            .iter()
            .map(|teacher| BitVec::repeat(false, teacher.timeslots_count))
            .collect();

        Self {
            room_busy,
            teacher_busy,
            remaining: (0..inst.lessons.len()).map(LessonId).collect(),
            placed: vec![],
            g_val: 0,
            h_val: 0,
            f_val: 0,
        }
    }
    pub fn is_goal(&self) -> bool {
        self.remaining.is_empty()
    }

    pub fn solve<A: SearchAlgorithm>(
        &mut self,
        algo: &A,
        inst: &Instance,
        lim: &Limits,
    ) -> Outcome {
        let start_time = Instant::now();
        let outcome = algo.run(inst, self, lim);
        let duration = start_time.elapsed();

        Outcome {
            millis: duration.as_millis(),
            ..outcome
        }
    }
}

pub struct Limits {
    pub time: Duration,
    pub mem: usize,
}

#[derive(Clone, Debug)]
pub struct Instance {
    pub rooms: Vec<Room>,
    pub teachers: Vec<Teacher>,
    pub lessons: Vec<Lesson>,
}


#[derive(Clone, Debug)]
pub struct Outcome {
    pub solved: bool,
    pub solution: Option<State>,
    pub generated: u64,
    pub max_in_mem: u64,
    pub millis: u128,
}

impl Outcome {
    pub fn new_solved(solution: State, generated: u64, max_in_mem: u64) -> Self {
        Outcome {
            solved: true,
            solution: Some(solution),
            generated,
            max_in_mem,
            millis: 0,
        }
    }

    pub fn new_unsolved(generated: u64, max_in_mem: u64) -> Self {
        Outcome {
            solved: false,
            solution: None,
            generated,
            max_in_mem,
            millis: 0,
        }
    }
}
