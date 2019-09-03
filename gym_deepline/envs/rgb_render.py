from PIL import Image, ImageDraw, ImageFont
from math import sin, cos, atan2, pi
import numpy as np
import os

white = (255, 255, 255)
green = (0, 128, 0)
black = (0, 0, 0)
orange = (255, 128, 0)
red = (255, 0, 0)
blue = (0, 0, 204)
abs_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
font = ImageFont.truetype(abs_path+"/Calibri.ttf", 14)


class MLGrid:
    def __init__(self, observation):
        self.cell_width = 180
        self.cell_height = 70
        self.level = observation.level
        self.rows = len(observation.grid)
        self.cols = len(observation.grid[0])
        self.width = self.cell_width*self.cols + 4
        self.height = self.cell_height*(self.rows+2) + 4
        self.image = Image.new("RGB", (self.width, self.height), black)
        self.draw = ImageDraw.Draw(self.image.resize((1088, 368)))
        self.agent = None

    def draw_arrow(self, x1, y1, x2, y2):
        r = 10
        betha = atan2(y1-y2, x2-x1)
        z1 = r*cos(betha - 0.3)
        z2 = r*sin(betha - 0.3)
        z3 = r*cos(pi/2 - betha - 0.3)
        z4 = r*sin(pi/2 - betha - 0.3)
        self.draw.line((x1, y1, x2, y2), fill=green)
        self.draw.line((x2-z1, y2+z2, x2, y2), fill=green)
        self.draw.line((x2-z4, y2+z3, x2, y2), fill=green)

    def generate_grid(self):
        for i in range(self.cols):
            for j in range(self.rows+2):
                if j == self.rows:
                    self.draw.rectangle((i * self.cell_width, j * self.cell_height, (i + 1) * self.cell_width, (j + 1) * self.cell_height), fill=black)
                else:
                    self.draw.rectangle((i * self.cell_width, j * self.cell_height, (i + 1) * self.cell_width, (j + 1) * self.cell_height), fill=white)
        self.agent = self.draw.rectangle((self.cell_width/2-10, self.cell_height/2-10, self.cell_width/2+10, self.cell_height/2+10), fill=green)
        return np.array(self.image.resize((1088, 368)))

    def reset(self, observation, action=None):
        self.image = Image.new("RGB", (self.width, self.height), black)
        self.draw = ImageDraw.Draw(self.image)

        for i in range(self.cols):
            for j in range(self.rows+2):
                if j == self.rows:
                    self.draw.rectangle((i * self.cell_width+3, j * self.cell_height+3, (i + 1) * self.cell_width+3, (j + 1) * self.cell_height+3), fill=black, outline=black)
                else:
                    self.draw.rectangle((i * self.cell_width+3, j * self.cell_height+3, (i + 1) * self.cell_width+3, (j + 1) * self.cell_height+3), fill=white, outline=black)

        self.agent = self.draw.ellipse((observation.cursor[1]*self.cell_width+self.cell_width/2-10, observation.cursor[0]*self.cell_height+self.cell_height/2-10, observation.cursor[1]*self.cell_width+self.cell_width/2+10, observation.cursor[0]*self.cell_height+self.cell_height/2+10), fill=green, outline=black)
        w, h = self.draw.textsize(observation.learning_job.name, font=font)
        self.draw.text((self.cell_width / 2 - w/2, self.rows*self.cell_height + self.cell_height / 2), fill=orange, text=observation.learning_job.name, font=font, align='center')
        w, h = self.draw.textsize('Reward: '+str(observation.last_reward), font=font)
        self.draw.text((3*self.cell_width / 2 - w/2, self.rows*self.cell_height + self.cell_height / 2), fill=orange, text='Reward: '+str(observation.last_reward), font=font, align='center')
        if not action==None:
            w, h = self.draw.textsize('Action: ', font=font)
            self.draw.text((5 * self.cell_width / 2 - w/2, self.rows * self.cell_height + self.cell_height / 2), fill=orange, text='Action: ' + str(action), font=font, align='center')
        for i in range(len(observation.grid)):
            for j in range(len(observation.grid[0])):
                if observation.grid[i][j] == 'BLANK' or observation.grid[i][j] == 'FINISH':
                    continue
                else:
                    name = observation.grid[i][j].primitive.name
                    w, h = self.draw.textsize(name + ' [' +str(observation.grid[i][j].index)+ ']', font=font)
                    self.draw.text((j * self.cell_width + self.cell_width / 2 - w/2,
                                            i * self.cell_height + self.cell_height / 2), fill=blue, text=name + ' [' +str(observation.grid[i][j].index)+ ']', font=font, align='center')
                    for input in observation.grid[i][j].input_indices:
                        step_idx = input[0]
                        if step_idx == 0:
                            self.draw_arrow(0, self.level*self.cell_height*0.65,
                                                    j * self.cell_width + self.cell_width / 2,
                                                    i * self.cell_height + self.cell_height * 0.75)
                        for l in range(len(observation.grid)):
                            for k in range(len(observation.grid[0])):
                                if observation.grid[l][k] == 'BLANK' or observation.grid[l][k] == 'FINISH':
                                    continue
                                elif observation.grid[l][k].index == step_idx:
                                    self.draw_arrow(k * self.cell_width + self.cell_width / 2,
                                            l * self.cell_height + self.cell_height*0.75, j * self.cell_width + self.cell_width / 2,
                                            i * self.cell_height + self.cell_height*0.75) # , arrow=tk.LAST

        if not len(observation.options_windows) == 0:
            for i in range(len(observation.options_windows[observation.window_index])):
                if observation.options_windows[observation.window_index][i] == -1:
                    continue
                if observation.options_windows[observation.window_index][i] == 'BLANK':
                    name = 'BLANK'
                    w, h = self.draw.textsize(name, font=font)
                    self.draw.text((i * self.cell_width + self.cell_width / 2 - w/2,
                                            (self.rows + 1) * self.cell_height + self.cell_height / 2), fill=red,
                                            text=name, font=font, align='center')
                elif observation.options_windows[observation.window_index][i] == 'FINISH':
                    name = 'FINISH'
                    w, h = self.draw.textsize(name, font=font)
                    self.draw.text((i * self.cell_width + self.cell_width / 2 - w/2,
                                            (self.rows + 1) * self.cell_height + self.cell_height / 2), fill=red,
                                            text=name, font=font, align='center')
                else:
                    name = observation.options_windows[observation.window_index][i].primitive.name
                    inpt = observation.options_windows[observation.window_index][i].input_indices
                    w, h = self.draw.textsize(name, font=font)
                    self.draw.text((i * self.cell_width + self.cell_width / 2 - w/2,
                                        (self.rows+1) * self.cell_height + self.cell_height / 2), align='center',  fill=red, text=name, font=font)
                    self.draw.text((i * self.cell_width + self.cell_width / 2 - w/2,
                                        (self.rows + 1) * self.cell_height + self.cell_height / 2 + 15), fill=red,
                                        text=str(inpt), font=font, align='center')

        # time.sleep(0.2)
        # self.canvas.postscript(file="tmp_canvas.eps",
        #                   colormode="color",
        #                   width=self.width,
        #                   height=self.height,
        #                   pagewidth=self.width - 1,
        #                   pageheight=self.height - 1)
        #
        # # read the postscript data
        # data = ski_io.imread("tmp_canvas.eps")
        #
        # # write a rasterized png file
        # ski_io.imsave("canvas_image.png", data)

        return np.array(self.image.resize((1088, 368)))
