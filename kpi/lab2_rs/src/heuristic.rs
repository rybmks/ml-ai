use crate::types::*;

pub fn heuristic(state: &State, inst: &Instance) -> u32 {
    let remaining_lessons_count = state.remaining.len() as u32; 

    let mut problematic_factor = 0;

    let mut needed_teachers = std::collections::HashSet::new();
    for lesson_id in &state.remaining {
        let lesson = &inst.lessons[lesson_id.0];
        for teacher_id in &lesson.candidate_teachers {
            needed_teachers.insert(teacher_id.0);
        }
    }

    // Compute the workload of teachers
    for &t_idx in needed_teachers.iter() {
        let teacher = &inst.teachers[t_idx];
        let busy_bits = state.teacher_busy[t_idx].count_ones();

        let total_teacher_slots = teacher.timeslots_count;

        // If a teacher is busy more than half of their working time
        if busy_bits > total_teacher_slots / 2 {
            problematic_factor += 1;
        }
    }

    let c_val = problematic_factor as u32;

    // Final formula: D(n) + C_load(n)
    remaining_lessons_count + c_val
}
