use crate::search::SearchAlgorithm;
use crate::types::*;
use rand::{Rng, seq::SliceRandom, thread_rng};
use std::time::Duration;

pub mod heuristic;
pub mod search;
pub mod types;

const NUM_EXPERIMENTS: usize = 20;

const MIN_LESSONS: usize = 3;
const MAX_LESSONS: usize = 4;
const LESSON_DURATION_MIN: usize = 1;
const LESSON_DURATION_MAX: usize = 2;

const MIN_TEACHERS: usize = 3;
const MAX_TEACHERS: usize = 5;
const T_SLOTS_MIN: usize = 8;
const T_SLOTS_MAX: usize = 12;

const NUM_ROOMS: usize = 2;
const ROOM_SLOTS_MIN: usize = 8;
const ROOM_SLOTS_MAX: usize = 10;

const TIME_LIMIT: Duration = Duration::from_secs(600);
const MEM_LIMIT: u64 = 5_000_000;

pub struct ExperimentResult {
    pub solved: bool,
    pub time_ms: u128,
    pub generated: u64,
    pub max_in_mem: u64,
}

pub fn generate_random_instance() -> Instance {
    let mut rng = thread_rng();

    let rooms: Vec<Room> = (0..NUM_ROOMS)
        .map(|i| {
            let slots = rng.gen_range(ROOM_SLOTS_MIN..=ROOM_SLOTS_MAX);
            Room::new(i, format!("Аудиторія {}", i + 1), slots)
        })
        .collect();

    let max_room_slots = rooms
        .iter()
        .map(|r| r.timeslots_count)
        .max()
        .unwrap_or(ROOM_SLOTS_MAX);

    let num_teachers = rng.gen_range(MIN_TEACHERS..=MAX_TEACHERS);
    let mut teachers = Vec::with_capacity(num_teachers);

    for i in 0..num_teachers {
        let max_possible_slots = max_room_slots.min(T_SLOTS_MAX);
        let slots = rng.gen_range(T_SLOTS_MIN..=max_possible_slots);

        teachers.push(Teacher {
            id: TeacherId(i),
            name: format!("Викладач {}", i + 1),
            timeslots_count: slots,
        });
    }

    let num_lessons = rng.gen_range(MIN_LESSONS..=MAX_LESSONS);
    let mut lessons = Vec::with_capacity(num_lessons);

    let teacher_ids: Vec<TeacherId> = teachers.iter().map(|t| t.id).collect();

    let mut main_teacher_index = 0;

    for i in 0..num_lessons {
        let duration = rng.gen_range(LESSON_DURATION_MIN..=LESSON_DURATION_MAX);

        let main_teacher_id = teacher_ids[main_teacher_index % teacher_ids.len()];
        main_teacher_index += 1;

        let mut candidate_teachers = vec![main_teacher_id];

        if rng.gen_bool(0.3) {
            let mut other_teachers: Vec<_> = teacher_ids
                .iter()
                .filter(|&&id| id != main_teacher_id)
                .copied()
                .collect();

            if !other_teachers.is_empty() {
                other_teachers.shuffle(&mut rng);
                candidate_teachers.push(other_teachers[0]);
            }
        }

        candidate_teachers.shuffle(&mut rng);

        lessons.push(Lesson {
            id: LessonId(i),
            name: format!("Лекція {}", i + 1),
            taken_timeslots: duration,
            candidate_teachers,
        });
    }

    lessons.shuffle(&mut rng);
    let lessons: Vec<Lesson> = lessons
        .into_iter()
        .enumerate()
        .map(|(id, mut l)| {
            l.id = LessonId(id);
            l
        })
        .collect();

    Instance {
        rooms,
        teachers,
        lessons,
    }
}

pub fn exp<A: SearchAlgorithm + Clone>(algo: A, limits: &Limits) -> Vec<ExperimentResult> {
    println!(
        "Запуск {} експериментів для {}",
        NUM_EXPERIMENTS,
        algo.name()
    );
    let mut results = Vec::with_capacity(NUM_EXPERIMENTS);

    for _i in 0..NUM_EXPERIMENTS {
        let inst_run = generate_random_instance();

        let mut initial_state = State::new(&inst_run);

        let outcome = initial_state.solve(&algo, &inst_run, limits);

        results.push(ExperimentResult {
            solved: outcome.solved,
            time_ms: outcome.millis,
            generated: outcome.generated,
            max_in_mem: outcome.max_in_mem,
        });

    }
    results
}

pub fn calculate_averages(results: &[ExperimentResult], algo_name: &str) {
    let successful_runs: Vec<&ExperimentResult> = results.iter().filter(|r| r.solved).collect();
    let num_successful = successful_runs.len();

    if num_successful == 0 {
        println!("\n Середні результати {}", algo_name);
        println!("Жоден експеримент не було успішно розв'язано.");
        return;
    }

    let avg_time =
        successful_runs.iter().map(|r| r.time_ms).sum::<u128>() as f64 / num_successful as f64;
    let avg_generated =
        successful_runs.iter().map(|r| r.generated).sum::<u64>() as f64 / num_successful as f64;
    let avg_max_mem =
        successful_runs.iter().map(|r| r.max_in_mem).sum::<u64>() as f64 / num_successful as f64;

    println!(
        "\n Середні",
    );
    println!("1. Середній час пошуку рішення: {:.4} с", avg_time / 1000.0);
    println!(
        "2. Середня кількість згенерованих станів: {:.2}",
        avg_generated
    );
    println!(
        "3. Середня кількість станів, що зберігаються в пам'яті: {:.2}",
        avg_max_mem
    );
    println!(
        "Рівень успішності: {:.2}%",
        (num_successful as f64 / NUM_EXPERIMENTS as f64) * 100.0
    );
}

pub fn print_schedule(instance: &Instance, state: &State) {
    println!("ЗНАЙДЕНИЙ РОЗКЛАД ({})", state.placed.len());

    let max_display_slots = instance
        .rooms
        .iter()
        .map(|r| r.timeslots_count)
        .max()
        .unwrap_or(12);
    let total_room_count = instance.rooms.len();

    let mut schedule_matrix = vec![vec!["[   ]".to_string(); max_display_slots]; total_room_count];

    for placement in &state.placed {
        let room_idx = placement.room.0;
        let item_label = format!("L{}T{}", placement.lesson.0 + 1, placement.teacher.0 + 1);

        for slot in placement.start..placement.start + placement.len {
            if room_idx < total_room_count && slot < max_display_slots {
                schedule_matrix[room_idx][slot] = format!("[{}]", item_label);
            }
        }
    }

    print!("| {:<10} |", "Час");
    for i in 0..max_display_slots {
        print!(" {:<5} |", format!("T{}", i + 1));
    }
    println!();
    println!("{}", "-".repeat(12 + (max_display_slots * 8)));

    for (r_idx, row) in schedule_matrix.iter().enumerate() {
        print!("| {:<10} |", instance.rooms[r_idx].name);
        for (c_idx, cell) in row.iter().enumerate() {
            if c_idx < instance.rooms[r_idx].timeslots_count {
                print!(" {:<5} |", cell);
            } else {
                print!(" {:<5} |", "[---]");
            }
        }
        println!();
    }
}

pub fn check_conflicts(instance: &Instance, state: &State) -> (bool, String) {
    let num_rooms = instance.rooms.len();
    let num_teachers = instance.teachers.len();
    let max_display_slots = instance
        .rooms
        .iter()
        .map(|r| r.timeslots_count)
        .max()
        .unwrap_or(1);

    let mut room_schedule = vec![vec![false; max_display_slots]; num_rooms];
    let mut teacher_schedule = vec![Vec::new(); num_teachers];

    for (t_idx, teacher) in instance.teachers.iter().enumerate() {
        teacher_schedule[t_idx] = vec![false; teacher.timeslots_count];
    }

    for placement in &state.placed {
        let r_idx = placement.room.0;
        let t_idx = placement.teacher.0;
        let len = placement.len;

        let room_limit = instance.rooms[r_idx].timeslots_count;
        let teacher_limit = instance.teachers[t_idx].timeslots_count;

        for slot in placement.start..placement.start + len {
            if slot >= teacher_limit {
                return (
                    false,
                    format!(
                        "Помилка: Лекція L{} виходить за межі робочого часу викладача T{} (Слот T{} > T{}).",
                        placement.lesson.0 + 1,
                        t_idx + 1,
                        slot + 1,
                        teacher_limit
                    ),
                );
            }

            if slot >= room_limit {
                return (
                    false,
                    format!(
                        "Помилка: Лекція L{} виходить за межі часу кімнати R{} (Слот T{} > T{}).",
                        placement.lesson.0 + 1,
                        r_idx + 1,
                        slot + 1,
                        room_limit
                    ),
                );
            }

            if room_schedule[r_idx][slot] {
                return (
                    false,
                    format!("Конфлікт кімнати R{} у слоті T{}.", r_idx + 1, slot + 1),
                );
            }
            room_schedule[r_idx][slot] = true;

            if teacher_schedule[t_idx][slot] {
                return (
                    false,
                    format!("Конфлікт викладача T{} у слоті T{}.", t_idx + 1, slot + 1),
                );
            }
            teacher_schedule[t_idx][slot] = true;
        }
    }

    if !state.remaining.is_empty() {
        return (
            false,
            format!(
                "Розклад не повний. Залишилося розмістити {} лекцій.",
                state.remaining.len()
            ),
        );
    }

    (
        true,
        "Конфліктів не виявлено. Розклад коректний і повний.".to_string(),
    )
}

pub fn run_experiments() {
    use crate::search::{astar::AStar, bfs::Bfs};

    let limits = Limits {
        time: TIME_LIMIT,
        mem: MEM_LIMIT as usize,
    };

    let bfs_algo = Bfs;
    let bfs_results = exp(bfs_algo, &limits);
    calculate_averages(&bfs_results, "BFS");

    println!("\n-------------------------------------------------------\n");

    let astar_algo = AStar;
    let astar_results = exp(astar_algo, &limits);
    calculate_averages(&astar_results, "A*");

    let guaranteed_limits = Limits {
        time: Duration::from_secs(900),
        mem: (MEM_LIMIT * 2) as usize,
    };

    let mut solution_outcome = None;
    let mut attempt_count = 0;
    let mut example_instance = generate_random_instance();
    let max_attempts = 10;

    while solution_outcome.is_none() && attempt_count < max_attempts {
        attempt_count += 1;
        example_instance = generate_random_instance();
        let mut initial_state = State::new(&example_instance);

        let outcome = initial_state.solve(&AStar, &example_instance, &guaranteed_limits);

        if outcome.solved {
            solution_outcome = outcome.solution;
        }
    }
}
