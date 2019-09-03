import matplotlib
matplotlib.use('Agg')
import tkinter as tk
import numpy as np
import random
import time


class MLGrid:
    def __init__(self, observation):
        self.cell_width = 180
        self.cell_height = 70
        self.level = observation.level
        self.rows = len(observation.grid)
        self.cols = len(observation.grid[0])
        self.width = self.cell_width*self.cols + 4
        self.height = self.cell_height*(self.rows+2) + 4
        self.root = tk
        self.canvas = self.root.Canvas(bg='black', height=self.height, width=self.width)
        self.agent = None

    def generate_grid(self):
        for i in range(self.cols):
            for j in range(self.rows+2):
                if j == self.rows:
                    self.canvas.create_rectangle(i * self.cell_width, j * self.cell_height, (i + 1) * self.cell_width, (j + 1) * self.cell_height, fill="black", width=3)
                else:
                    self.canvas.create_rectangle(i * self.cell_width, j * self.cell_height, (i + 1) * self.cell_width, (j + 1) * self.cell_height, fill="white", width=3)
        self.agent = self.canvas.create_oval(self.cell_width/2-10, self.cell_height/2-10, self.cell_width/2+10, self.cell_height/2+10, fill="green", width=2)
        self.canvas.pack()

    def reset(self, observation, action=None):
        self.canvas.delete('all')

        for i in range(self.cols):
            for j in range(self.rows+2):
                if j == self.rows:
                    self.canvas.create_rectangle(i * self.cell_width+3, j * self.cell_height+3, (i + 1) * self.cell_width+3, (j + 1) * self.cell_height+3, fill="black", width=3)
                else:
                    self.canvas.create_rectangle(i * self.cell_width+3, j * self.cell_height+3, (i + 1) * self.cell_width+3, (j + 1) * self.cell_height+3, fill="white", width=3)

        self.agent = self.canvas.create_oval(observation.cursor[1]*self.cell_width+self.cell_width/2-10, observation.cursor[0]*self.cell_height+self.cell_height/2-10, observation.cursor[1]*self.cell_width+self.cell_width/2+10, observation.cursor[0]*self.cell_height+self.cell_height/2+10, fill="green", width=2)
        self.canvas.create_text(self.cell_width / 2, self.rows*self.cell_height + self.cell_height / 2, fill="orange", text=observation.learning_job.name)
        self.canvas.create_text(3*self.cell_width / 2, self.rows*self.cell_height + self.cell_height / 2, fill="orange", text='Reward: '+str(observation.last_reward))
        if not action==None:
            self.canvas.create_text(5 * self.cell_width / 2, self.rows * self.cell_height + self.cell_height / 2, fill="orange", text='Action: ' + str(action))
        for i in range(len(observation.grid)):
            for j in range(len(observation.grid[0])):
                if observation.grid[i][j] == 'BLANK' or observation.grid[i][j] == 'FINISH':
                    continue
                else:
                    name = observation.grid[i][j].primitive.name
                    self.canvas.create_text(j * self.cell_width + self.cell_width / 2,
                                            i * self.cell_height + self.cell_height / 2, fill="blue", text=name + ' [' +str(observation.grid[i][j].index)+ ']', font='Calibri 10 bold')
                    for input in observation.grid[i][j].input_indices:
                        step_idx = input[0]
                        if step_idx == 0:
                            self.canvas.create_line(0, self.level*self.cell_height*0.65,
                                                    j * self.cell_width + self.cell_width / 2,
                                                    i * self.cell_height + self.cell_height * 0.75, arrow=tk.LAST,
                                                    fill='green')
                        for l in range(len(observation.grid)):
                            for k in range(len(observation.grid[0])):
                                if observation.grid[l][k] == 'BLANK' or observation.grid[l][k] == 'FINISH':
                                    continue
                                elif observation.grid[l][k].index == step_idx:
                                    self.canvas.create_line(k * self.cell_width + self.cell_width / 2,
                                            l * self.cell_height + self.cell_height*0.75, j * self.cell_width + self.cell_width / 2,
                                            i * self.cell_height + self.cell_height*0.75, arrow=tk.LAST, fill='green')

        if not len(observation.options_windows) == 0:
            for i in range(len(observation.options_windows[observation.window_index])):
                if observation.options_windows[observation.window_index][i] == -1:
                    continue
                if observation.options_windows[observation.window_index][i] == 'BLANK':
                    name = 'BLANK'
                    self.canvas.create_text(i * self.cell_width + self.cell_width / 2,
                                            (self.rows + 1) * self.cell_height + self.cell_height / 2, fill="red",
                                            text=name)
                elif observation.options_windows[observation.window_index][i] == 'FINISH':
                    name = 'FINISH'
                    self.canvas.create_text(i * self.cell_width + self.cell_width / 2,
                                            (self.rows + 1) * self.cell_height + self.cell_height / 2, fill="red",
                                            text=name)
                else:
                    name = observation.options_windows[observation.window_index][i].primitive.name
                    inpt = observation.options_windows[observation.window_index][i].input_indices
                    self.canvas.create_text(i * self.cell_width + self.cell_width / 2,
                                        (self.rows+1) * self.cell_height + self.cell_height / 2, fill="red", text=name)
                    self.canvas.create_text(i * self.cell_width + self.cell_width / 2,
                                        (self.rows + 1) * self.cell_height + self.cell_height / 2 + 15, fill="red",
                                        text=str(inpt))
        self.canvas.pack()
        self.canvas.update()
        # time.sleep(0.2)
        # if observation.last_reward > 0:
        #     time.sleep(1.5)
