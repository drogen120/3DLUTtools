from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.graphics import Color, Rectangle
from random import random as r
from functools import partial
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.properties import ObjectProperty

import os
import csv
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# AxisList = ["Red", "Green", "Blue"]
AxisList = ["Blue", "Green", "Red"] #Because of the reshape, the axis is different from lut doc
AxisPlotDic = {"Red" : 0, "Green" : 1, "Blue" : 2}
class Draw3DSurface(object):
    def __init__(self, step=0.030304):
        # self.lut_table = np.zeros((33, 33, 33, 3), dtype = np.float)
        self.X = np.arange(0, 1, step)
        self.Y = np.arange(0, 1, step)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
    def plot(self, Z , plot_axis):
        fig = plt.figure()

        ax = fig.gca(projection='3d')
        ax.plot_surface(self.X, self.Y, Z, rstride=1, cstride=1, alpha=0.3)
        cset = ax.contour(self.X, self.Y, Z, zdir='z', cmap=cm.coolwarm)
        cset = ax.contour(self.X, self.Y, Z, zdir='x', cmap=cm.coolwarm)
        cset = ax.contour(self.X, self.Y, Z, zdir='y', cmap=cm.coolwarm)

        # plt.subplot(221)

        ax.set_xlabel('X')
        ax.set_xlim(0, 1)
        ax.set_ylabel('Y')
        ax.set_ylim(0, 1)
        ax.set_zlabel(plot_axis)
        ax.set_zlim(0, 1)

        plt.show()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class LUTData(object):
    def __init__(self):
        # self.lut_table = np.zeros((33, 33, 33, 3), dtype = np.float)
        self.lut_table = None

    def set_lut(self, lut):
        self.lut_table = lut

    def is_empty(self):
        if self.lut_table == None:
            return True
        else:
            return False

    def get_lut_layer(self, axis=0, index=0):
        if axis == 0:
            return self.lut_table[index, :, :, :]
        elif axis == 1:
            return self.lut_table[:, index, :, :]
        elif axis == 2:
            return self.lut_table[:, :, index, :]

class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    lut_data1 = LUTData()
    lut_data2 = LUTData()
    file_index = 1

    def readLUT(self, path, filename):
        with open(os.path.join(path, filename[0]), 'rb') as lutfile:
            LUT_table = []
            skip_lines = 9
            datareader = csv.reader(lutfile)
            for i in range(skip_lines):
                datareader.next()

            for row in datareader:
                item = map(float, row[0].split(" "))
                LUT_table.append(item)
            LUT_table = np.array(LUT_table)
            # print LUT_table.shape
            LUT_table = LUT_table.reshape((33, 33, 33, 3))
        return LUT_table

    def set_file_index(self, index):
        self.file_index = index

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        print (path, filename)
        lut = self.readLUT(path, filename)
        if self.file_index == 1:
            self.lut_data1.set_lut(lut)
        else:
            self.lut_data2.set_lut(lut)

        self.dismiss_popup()

    def save(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.text_input.text)

        self.dismiss_popup()


class LUTMain(Widget):
    pass

class LUTtoolsApp(App):

    def add_rects(self, wid, count, *largs):
        with wid.canvas:
            for x in range(count):
                Color(r(), 1, 1, mode='hsv')
                Rectangle(pos=(r() * wid.width + wid.x,
                               r() * wid.height + wid.y), size=(20, 20))

    def show_lut_layer(self, wid, *largs):
        with wid.canvas:
            if self.fileroot.lut_data1.is_empty() == False:
                lut_layer = self.fileroot.lut_data1.get_lut_layer(self.axis, self.layer_index)
                # print lut_layer.shape
                lut_layer = np.reshape(lut_layer, (-1,3))
                # print lut_layer.shape
                for i in range(lut_layer.shape[0]):
                    x = i % 33
                    y = i / 33
                    Color(lut_layer[i,0], lut_layer[i,1], lut_layer[i,2])
                    Rectangle(pos=(x * 35 + wid.x + 20,
                                   y * 20 + wid.y + 20), size=(10, 15))

            if self.fileroot.lut_data2.is_empty() == False:
                lut_layer = self.fileroot.lut_data2.get_lut_layer(self.axis, self.layer_index)
                # print lut_layer.shape
                lut_layer = np.reshape(lut_layer, (-1,3))
                # print lut_layer.shape
                for i in range(lut_layer.shape[0]):
                    x = i % 33
                    y = i / 33
                    Color(lut_layer[i,0], lut_layer[i,1], lut_layer[i,2])
                    Rectangle(pos=(x * 35 + wid.x + 32,
                                   y * 20 + wid.y + 20), size=(10, 15))

    def load_lut(self, wid, file_index, *largs):
        if file_index == 1:
            load_file = 1
        else:
            load_file = 2
        self.fileroot.set_file_index(load_file)
        self.fileroot.show_load()

    def change_axis(self, wid, *largs):
        self.axis = (self.axis + 1) % len(AxisList)
        self.update_label()
        self.show_lut_layer(wid)

    def show_3D_plot(self, wid, *largs):
        lut_layer = self.fileroot.lut_data1.get_lut_layer(self.axis, self.layer_index)
        for key, value in AxisPlotDic.iteritems():
            if key != AxisList[self.axis]:
                self.plot3d.plot(lut_layer[:,:,value], key)

    def reset_rects(self, wid, *largs):
        pass

    def OnSliderValueChange(self, wid, instance, value):
        # self.label.text = pattern.format(AxisList[self.axis], value)
        self.layer_index = int(value)
        self.update_label()
        self.show_lut_layer(wid)

    def update_label(self):
        self.label.text = self.label_pattern.format(AxisList[self.axis], self.layer_index)

    def build(self):
        wid = Widget(size_hint=(0.95, 1))
        slider = Slider(min=0, max=32, value=0, value_track=True, orientation='vertical',
        step=1.0, value_track_color=[1, 0, 0, 1], size_hint=(0.2, 1))
        self.label_pattern = "Axis {} : {}"
        self.fileroot = Root()
        self.axis = 2
        self.layer_index = 0
        self.load_file = 0
        self.plot3d = Draw3DSurface()
        upper_layout = BoxLayout()
        upper_layout.add_widget(wid)
        upper_layout.add_widget(slider)

        self.label = Label(text=self.label_pattern.format(AxisList[self.axis], 0))

        btn_load_lut1 = Button(text='Load LUT 1',
                            on_press=partial(self.load_lut, wid, 1))

        btn_load_lut2 = Button(text='Load LUT 2',
                            on_press=partial(self.load_lut, wid, 2))

        btn_showlayer = Button(text='Show LUT Layer',
                            on_press=partial(self.show_lut_layer, wid))

        btn_showplot = Button(text='Show 3D Plot',
                            on_press=partial(self.show_3D_plot, wid))

        btn_double = Button(text='Change Axis',
                            on_press=partial(self.change_axis, wid))

        btn_reset = Button(text='Reset',
                           on_press=partial(self.reset_rects, wid))

        layout = BoxLayout(size_hint=(1, None), height=50)
        layout.add_widget(btn_load_lut1)
        layout.add_widget(btn_load_lut2)
        layout.add_widget(btn_showlayer)
        layout.add_widget(btn_showplot)
        layout.add_widget(btn_double)
        layout.add_widget(btn_reset)
        layout.add_widget(self.label)

        root = BoxLayout(orientation='vertical')
        root.add_widget(upper_layout)
        root.add_widget(layout)
        slider.bind(value=partial(self.OnSliderValueChange, wid))

        return root

Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

if __name__ == '__main__':
    LUTtoolsApp().run()
