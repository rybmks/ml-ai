use std::time::Duration;

use lab2::{
    print_schedule, run_experiments,
    search::{astar::AStar, bfs::Bfs},
    types::*,
};

fn main() {
    let rooms = vec![
        Room::new(0, String::from("A101"), 6),
        Room::new(1, String::from("B202"), 8),
    ];

    let teacher1 = Teacher {
        id: TeacherId(0),
        name: "Викладач А".into(),
        timeslots_count: 10,
    };
    let teacher2 = Teacher {
        id: TeacherId(1),
        name: "Викладач Б".into(),
        timeslots_count: 10,
    };
    let teacher3 = Teacher {
        id: TeacherId(2),
        name: "Викладач В".into(),
        timeslots_count: 10,
    };
    let teacher4 = Teacher {
        id: TeacherId(3),
        name: "Викладач Г".into(),
        timeslots_count: 10,
    };
    let teachers = vec![teacher1, teacher2, teacher3, teacher4];

    let lessons = vec![
        Lesson {
            id: LessonId(0),
            name: String::from("Математика"),
            taken_timeslots: 2,
            candidate_teachers: vec![TeacherId(0)],
        },
        Lesson {
            id: LessonId(1),
            name: String::from("Програмування"),
            taken_timeslots: 2,
            candidate_teachers: vec![TeacherId(1)],
        },
        Lesson {
            id: LessonId(2),
            name: String::from("Алгоритми"),
            taken_timeslots: 2,
            candidate_teachers: vec![TeacherId(2)],
        },
        Lesson {
            id: LessonId(3),
            name: String::from("Бази даних"),
            taken_timeslots: 2,
            candidate_teachers: vec![TeacherId(3)],
        },
    ];

    let inst = Instance {
        rooms,
        teachers,
        lessons,
    };

    let lim = Limits {
        time: Duration::from_secs(600),
        mem: 4_000_000,
    };

    let algo_bfs = Bfs;
    let mut state_bfs = State::new(&inst);

    let outcome_bfs = state_bfs.solve(&algo_bfs, &inst, &lim);

    println!("\nРЕЗУЛЬТАТ BFS");
    if let Some(solution) = outcome_bfs.solution {
        println!("Розв'язок знайдено! Час: {} мс", outcome_bfs.millis);
        lab2::check_conflicts(&inst, &solution);
        print_schedule(&inst, &solution);
    } else {
        println!("Рішення BFS не знайдено в межах обмежень.");
    }

    let algo_astar = AStar;
    let mut state_astar = State::new(&inst);

    let outcome_astar = state_astar.solve(&algo_astar, &inst, &lim);

    println!("\nРЕЗУЛЬТАТ A*");
    if let Some(solution) = outcome_astar.solution {
        println!("Розв'язок знайдено! Час: {} мс", outcome_astar.millis);
        lab2::check_conflicts(&inst, &solution);
        print_schedule(&inst, &solution);
    } else {
        println!("Рішення A* не знайдено в межах обмежень.");
    }

    run_experiments();
}
