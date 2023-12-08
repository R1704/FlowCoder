import deepsynth.dsl as dsl
from deepsynth.DSL.list import semantics, primitive_types
from deepsynth.model_loader import build_dreamcoder_intlist_model
from deepsynth.dreamcoder_dataset_loader import load_tasks, filter_tasks_for_model
from deepsynth.type_system import Arrow, List, INT

from flowcoder.config import *

import random
import numpy as np
from itertools import chain


class Data:

    def __init__(self, max_program_depth=6, shuffle_tasks=False, variable_batch=True, n_tasks=95, train_ratio=0.5, seed=42):
        self.max_program_depth = max_program_depth
        self.shuffle_tasks = shuffle_tasks
        self.n_tasks = min(n_tasks, 95)
        self.variable_batch = variable_batch
        self.train_ratio = train_ratio
        self.n_train_tasks = int(np.ceil(n_tasks * train_ratio))
        self.n_test_tasks = self.n_tasks - self.n_train_tasks

        self.type_request = Arrow(List(INT), List(INT))
        self.dsl = dsl.DSL(semantics, primitive_types)
        self.cfg = self.dsl.DSL_to_CFG(
            self.type_request,
            max_program_depth=max_program_depth
            )

        self.nb_examples_max = 15  # from DreamCoder dataset
        self.size_max = 10
        nb_arguments_max = 1
        self.lexicon = [x for x in range(-30, 30)]

        self.arguments = {self.type_request: self.type_request.arguments()}
        self.tasks, self.dataset_size = self.load_dreamcoder_tasks()

        # for control over tasks
        rng = random.Random(seed)
        if self.shuffle_tasks:
            rng.shuffle(self.tasks)
        self.tasks = self.tasks[:self.n_tasks]

        self.train_tasks, self.test_tasks = self.split_dataset()

        print('\n------ Train tasks ------')
        for train_task in self.train_tasks:
            name, _ = train_task
            print(name)

        print('\n------Test tasks ------')
        for test_task in self.test_tasks:
            name, _ = test_task
            print(name)
        print()

        self.train_task_generator = self.make_task_generator(self.train_tasks)
        self.test_task_generator = self.make_task_generator(self.test_tasks)

    def sample_input(self):
        nb_IOs = random.randint(1, self.nb_examples_max)
        inputs = [random.sample(range(max(self.lexicon)-1), random.randint(0, self.size_max)) for _ in range(nb_IOs)]  # the empirical data is in range [0, 29]
        return inputs

    def load_dreamcoder_tasks(self):
        # Load tasks
        tasks = load_tasks(DREAMCODER_DATASET_PATH)
        print("Loaded", len(tasks), "tasks")

        # Filter tasks
        _, _, rules_predictor = build_dreamcoder_intlist_model(max_program_depth=self.max_program_depth)
        tasks = filter_tasks_for_model(tasks, rules_predictor)
        with open("task_names.txt", "r") as file:
            task_names = file.read().splitlines()
        # Create a dictionary to map task names to their corresponding tasks
        task_dict = {name: task for name, task in tasks}
        # Filter the tasks based on whether they are in the task_names list and maintain the order
        tasks = [(name, task_dict[name]) for name in task_names if name in task_dict]
        print("Remaining tasks after filter:", len(tasks), "tasks")

        # Format tasks array
        dataset_size = len(tasks)
        all_tasks = []
        for name, examples in tasks:
            ex = [([i[0]], o) for i, o in examples]
            all_tasks.append((name, ex))
        return all_tasks, dataset_size

    def get_io_batch(self, batch_size, train=True, all_tasks=False):
        # Make task generator
        task_generator = self.train_task_generator if train else self.test_task_generator
        if all_tasks:
            task_generator = chain(self.train_task_generator, self.test_task_generator)

        # Make IO batches
        batch_IOs, batch_program_names = [], []
        task = None
        for _ in range(batch_size):
            try:
                if task is None or self.variable_batch:
                    task = next(task_generator)
                name, ios = task[0], task[1]
                print(name)
                batch_program_names.append(name)
                batch_IOs.append(ios)
            except StopIteration:
                break
        return batch_IOs, batch_program_names

    def make_task_generator(self, tasks):
        for task in tasks:
            yield task

    # Resets the task generators to their initial states.
    def reset_task_generators(self, train=True, test=True):
        if train:
            self.train_task_generator = self.make_task_generator(self.train_tasks)
        if test:
            self.test_task_generator = self.make_task_generator(self.test_tasks)

    def split_dataset(self):
        return self.tasks[:self.n_train_tasks], self.tasks[self.n_train_tasks:]
