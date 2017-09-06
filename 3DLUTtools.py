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
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.tabbedpanel import TabbedPanelHeader
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.animation import Animation
from kivy.properties import NumericProperty
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock, mainthread
from kivy.uix.textinput import TextInput
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.uix.togglebutton import ToggleButton

import os
import csv
import threading
from PIL import Image as PILImage
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from math import sin
from utils import NumericStringParser

AxisList = ["Red", "Green", "Blue"]
# AxisList = ["Blue", "Green", "Red"] #Because of the reshape, the axis is different from lut doc
AxisPlotDic = {"Red" : 0, "Green" : 1, "Blue" : 2}

class PopupBox(Popup):
    pop_up_text = ObjectProperty()
    def update_pop_up_text(self, p_message):
        self.pop_up_text.text = p_message

class Draw3DSurface(object):
    def __init__(self, step=0.030304):
        # self.lut_table = np.zeros((33, 33, 33, 3), dtype = np.float)
        self.X = np.arange(0, 1, step)
        self.Y = np.arange(0, 1, step)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
    def plot(self, Z1, Z2, plot_axis):
        if Z1 is not None and Z2 is not None:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot_surface(self.X, self.Y, Z1, rstride=1, cstride=1, alpha=0.3)
            cset = ax.contour(self.X, self.Y, Z1, zdir='z', cmap=cm.coolwarm)
            cset = ax.contour(self.X, self.Y, Z1, zdir='x', cmap=cm.coolwarm)
            cset = ax.contour(self.X, self.Y, Z1, zdir='y', cmap=cm.coolwarm)
            ax.set_xlabel('X')
            ax.set_xlim(0, 1)
            ax.set_ylabel('Y')
            ax.set_ylim(0, 1)
            ax.set_zlabel(plot_axis)
            ax.set_zlim(0, 1)

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.plot_surface(self.X, self.Y, Z2, rstride=1, cstride=1, alpha=0.3)
            cset = ax.contour(self.X, self.Y, Z2, zdir='z', cmap=cm.coolwarm)
            cset = ax.contour(self.X, self.Y, Z2, zdir='x', cmap=cm.coolwarm)
            cset = ax.contour(self.X, self.Y, Z2, zdir='y', cmap=cm.coolwarm)
            ax.set_xlabel('X')
            ax.set_xlim(0, 1)
            ax.set_ylabel('Y')
            ax.set_ylim(0, 1)
            ax.set_zlabel(plot_axis)
            ax.set_zlim(0, 1)

            plt.show()
        elif Z1 is not None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(self.X, self.Y, Z1, rstride=1, cstride=1, alpha=0.3)
            cset = ax.contour(self.X, self.Y, Z1, zdir='z', cmap=cm.coolwarm)
            cset = ax.contour(self.X, self.Y, Z1, zdir='x', cmap=cm.coolwarm)
            cset = ax.contour(self.X, self.Y, Z1, zdir='y', cmap=cm.coolwarm)
            ax.set_xlabel('X')
            ax.set_xlim(0, 1)
            ax.set_ylabel('Y')
            ax.set_ylim(0, 1)
            ax.set_zlabel(plot_axis)
            ax.set_zlim(0, 1)
            plt.show()
        elif Z2 is not None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(self.X, self.Y, Z2, rstride=1, cstride=1, alpha=0.3)
            cset = ax.contour(self.X, self.Y, Z2, zdir='z', cmap=cm.coolwarm)
            cset = ax.contour(self.X, self.Y, Z2, zdir='x', cmap=cm.coolwarm)
            cset = ax.contour(self.X, self.Y, Z2, zdir='y', cmap=cm.coolwarm)
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

    def get_lut(self):
        return np.copy(self.lut_table)

    def get_lut_by_channel(self, channel_index):
        return np.copy(self.lut_table[:,:,:,channel_index])

    def set_lut_by_channel(self, channel_index, lut_channel):
        self.lut_table[:,:,:,channel_index] = lut_channel

    def get_lut_list(self):
        output_lut_table = self.lut_table.transpose((2,1,0,3))
        output_lut_table = np.copy(np.reshape(output_lut_table, (-1,3)))
        output_lut_table = output_lut_table.tolist()
        return output_lut_table

    def make_identy_lut_list(self):
        x = np.linspace(0, 1.0, 33)
        y = np.linspace(0, 1.0, 33)
        z = np.linspace(0, 1.0, 33)
        a_x, a_y, a_z = np.meshgrid(x,y,z)
        identy_array = np.stack((a_x,a_y,a_z), axis = 3)
        identy_array = identy_array.transpose((2, 0, 1, 3))
        identy_array = np.reshape(identy_array, (-1, 3))
        identy_list = identy_array.tolist()
        return identy_list

    def is_empty(self):
        if self.lut_table is None:
            return True
        else:
            return False

    def get_lut_layer(self, axis=0, index=0):
        if axis == 0:
            result = self.lut_table[index, :, :, :]
        elif axis == 1:
            result = self.lut_table[:, index, :, :]
        elif axis == 2:
            result = self.lut_table[:, :, index, :]

        return np.copy(result)

    def get_edit_lutdata(self, keep_axis, layer_index, edit_axis, slider_axis, slider_value):
        if keep_axis == 0:
            if slider_axis == 2:
                result = self.lut_table[layer_index, :, slider_value, edit_axis]
            else:
                result = self.lut_table[layer_index, slider_value, :, edit_axis]
        elif keep_axis == 1:
            if slider_axis == 2:
                result = self.lut_table[:, layer_index, slider_value, edit_axis]
            else:
                result = self.lut_table[slider_value, layer_index, :, edit_axis]
        elif keep_axis == 2:
            if slider_axis == 1:
                result = self.lut_table[:, slider_value, layer_index, edit_axis]
            else:
                result = self.lut_table[slider_value, :, layer_index, edit_axis]

        return np.copy(result)

    def get_edit_preview_colordata(self, keep_axis, layer_index, slider_axis, slider_value):
        if keep_axis == 0:
            if slider_axis == 2:
                result = self.lut_table[layer_index, :, slider_value, :]
            else:
                result = self.lut_table[layer_index, slider_value, :, :]
        elif keep_axis == 1:
            if slider_axis == 2:
                result = self.lut_table[:, layer_index, slider_value, :]
            else:
                result = self.lut_table[slider_value, layer_index, :, :]
        elif keep_axis == 2:
            if slider_axis == 1:
                result = self.lut_table[:, slider_value, layer_index, :]
            else:
                result = self.lut_table[slider_value, :, layer_index, :]

        return np.copy(result)

    def set_edit_lutdata(self, keep_axis, layer_index, edit_axis, slider_axis, slider_value, new_data):
        if keep_axis == 0:
            if slider_axis == 2:
                self.lut_table[layer_index, :, slider_value, edit_axis] = new_data
            else:
                self.lut_table[layer_index, slider_value, :, edit_axis] = new_data
        elif keep_axis == 1:
            if slider_axis == 2:
                self.lut_table[:, layer_index, slider_value, edit_axis] = new_data
            else:
                self.lut_table[slider_value, layer_index, :, edit_axis] = new_data
        elif keep_axis == 2:
            if slider_axis == 1:
                self.lut_table[:, slider_value, layer_index, edit_axis] = new_data
            else:
                self.lut_table[slider_value, :, layer_index, edit_axis] = new_data

    def applyLUT(self, img):

        if self.is_empty() == True:
            return None
        x = np.linspace(0, 1.0, 33)
        y = np.linspace(0, 1.0, 33)
        z = np.linspace(0, 1.0, 33)
        lut = self.get_lut()
        # lut = lut.transpose((2,1,0,3))

        img = img / 255.0

        interpolating_function = RegularGridInterpolator((x, y, z), lut)
        count = 0
        result_img = np.zeros_like(img, dtype=np.float)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                pts = img[i, j, :]
                result_img[i, j, :] = interpolating_function(pts)

        return np.uint8(result_img * 255.0)

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
            # skip_lines = 9
            datareader = csv.reader(lutfile)
            for head_line in datareader:
                if len(head_line) > 0:
                    if head_line[0].find("LUT_3D_SIZE") > -1:
                        datareader.next()
                        datareader.next()
                        break
            # for i in range(skip_lines):
            #     datareader.next()

            for row in datareader:
                item = map(float, row[0].split(" "))
                LUT_table.append(item)
            LUT_table = np.array(LUT_table)
            # print LUT_table.shape
            LUT_table = LUT_table.reshape((33, 33, 33, 3))
            LUT_table = LUT_table.transpose((2,1,0,3))
        return LUT_table

    def readDat(self, path, filename):
        with open(os.path.join(path, filename[0]), 'rb') as datfile:
            Dat_table = []
            # skip_lines = 9
            datareader = csv.reader(datfile)
            for head_line in datareader:
                if len(head_line) > 0:
                    if head_line[0].find("3DLUTSIZE") > -1:
                        datareader.next()
                        break
            # for i in range(skip_lines):
            #     datareader.next()

            for row in datareader:
                item = map(float, row[0].split(" "))
                Dat_table.append(item)
            Dat_table = np.array(Dat_table)
            # print Dat_table.shape
            Dat_table = Dat_table.reshape((33, 33, 33, 3))
            # Dat_table = Dat_table.transpose((2,1,0,3))
        return Dat_table

    def saveLUT(self, path, filename, LUT_data):
        with open(os.path.join(path, filename), 'w') as lutfile:
            datawriter = csv.writer(lutfile, delimiter=' ')
            datawriter.writerow(["LUT_3D_SIZE","33"])
            datawriter.writerow(["LUT_3D_INPUT_RANGE", "0.0000000000", "1.0000000000"])
            datawriter.writerow([])
            output_list = LUT_data.get_lut_list()
            # output_list = LUT_data.make_identy_lut_list()
            for row in output_list:
                datawriter.writerow(row)


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
        if filename[0].split(".")[-1].lower() == "cube":
            lut = self.readLUT(path, filename)
        elif filename[0].split(".")[-1].lower() == "dat":
            lut = self.readDat(path, filename)
        if self.file_index == 1:
            self.lut_data1.set_lut(lut)
        else:
            self.lut_data2.set_lut(lut)

        self.dismiss_popup()

    def save(self, path, filename):
        self.saveLUT(path, filename, self.lut_data1)

        self.dismiss_popup()

class LUTMain(Widget):
    pass

class LUTtoolsApp(App):

    def show_popup(self):
        self.pop_up = Factory.PopupBox()
        self.pop_up.update_pop_up_text('Applying the LUT...')
        self.pop_up.open()

    def add_rects(self, wid, count, *largs):
        with wid.canvas:
            for x in range(count):
                Color(r(), 1, 1, mode='hsv')
                Rectangle(pos=(r() * wid.width + wid.x,
                               r() * wid.height + wid.y), size=(20, 20))

    def show_lut_layer(self, wid, *largs):
        with wid.canvas:
            wid.canvas.clear()
            if self.fileroot.lut_data1.is_empty() == False and self.fileroot.lut_data2.is_empty() == True:
                # self.update_edit_panel()
                lut_layer = self.fileroot.lut_data1.get_lut_layer(self.axis, self.layer_index)
                # print lut_layer.shape
                lut_layer = np.reshape(lut_layer, (-1,3))
                # print lut_layer.shape
                for i in range(lut_layer.shape[0]):
                    x = i % 33
                    y = i / 33
                    Color(lut_layer[i,0], lut_layer[i,1], lut_layer[i,2])
                    Rectangle(pos=(x * wid.width / 33.0 + wid.x + 15,
                                   y * wid.height / 34.0 + wid.y + 15), size=(wid.width / 35.0 , wid.height / 35.0))

            if self.fileroot.lut_data1.is_empty() == False and self.fileroot.lut_data2.is_empty() == False:
                lut_layer = self.fileroot.lut_data1.get_lut_layer(self.axis, self.layer_index)
                # print lut_layer.shape
                lut_layer = np.reshape(lut_layer, (-1,3))
                # print lut_layer.shape
                for i in range(lut_layer.shape[0]):
                    x = i % 33
                    y = i / 33
                    Color(lut_layer[i,0], lut_layer[i,1], lut_layer[i,2])
                    Rectangle(pos=(x * wid.width / 32.0 + wid.x + 5,
                                   y * wid.height / 34.0 + wid.y + 15), size=(wid.width / 71.0 , wid.height / 35.0))
                lut_layer = self.fileroot.lut_data2.get_lut_layer(self.axis, self.layer_index)
                # print lut_layer.shape
                lut_layer = np.reshape(lut_layer, (-1,3))
                # print lut_layer.shape
                for i in range(lut_layer.shape[0]):
                    x = i % 33
                    y = i / 33
                    Color(lut_layer[i,0], lut_layer[i,1], lut_layer[i,2])
                    Rectangle(pos=(x * wid.width / 32.0 + wid.x + 5 +wid.width / 70.0,
                                   y * wid.height / 34.0 + wid.y + 15), size=(wid.width / 71.0 , wid.height / 35.0))

    def update_edit_panel(self, c_wid):
        # lut_layer = self.fileroot.lut_data1.get_lut_layer(self.axis, self.layer_index)
        color_data = self.fileroot.lut_data1.get_edit_lutdata(self.axis, self.layer_index, self.edit_axis, self.slider_axis, self.slider_value)
        self.color_data_list = color_data.tolist()
        for sld, value in zip(self.slider_list, self.color_data_list):
            sld.value = value

        self.color_preview_data = self.fileroot.lut_data1.get_edit_preview_colordata(self.axis, self.layer_index,
                self.slider_axis, self.slider_value)
        self.show_edit_preview_color(c_wid, self.color_preview_data)
        self.update_edit_label()

    def show_pre_image(self, img_wid, *largs):

        im = PILImage.open("./img/wql_result.jpeg")
        im = im.transpose(PILImage.FLIP_TOP_BOTTOM)
        self.img_width, self.img_height = im.size
        # print im.size
        # print img_wid.height
        self.img_array = np.array(im)
        # print self.img_array
        temp_array = np.copy(self.img_array)
        texture = Texture.create(size=(500, 350), colorfmt="rgb")
        data = temp_array.tostring()
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")

        with img_wid.canvas:
            Rectangle(texture=texture, pos = (img_wid.center_x / 2.0 - self.img_width / 2.0,
                img_wid.center_y - self.img_height / 2.0),size=(500, 350))

    def apply_lut_image(self, img_wid):
        self.pb.value = 300
        self.lut_img_array = self.fileroot.lut_data1.applyLUT(np.copy(self.img_array))
        self.pb.value = 1000
        # print self.lut_img_array
        # temp = np.uint8(self.lut_img_array)
        self.update_img_widget(img_wid)
        # self.pop_up.dismiss()

    def onApplyLUTClick(self, img_wid, *largs):
        self.show_popup()
        mythread = threading.Thread(target=self.apply_lut_image, args=(img_wid,))
        mythread.start()

    @mainthread
    def update_img_widget(self, img_wid):
        self.pop_up.dismiss()
        temp = np.copy(self.lut_img_array)
        texture = Texture.create(size=(500, 350), colorfmt="rgb")
        data = temp.tostring()
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")

        with img_wid.canvas:
            # img_wid.canvas.clear()
            Rectangle(texture=texture, pos = (img_wid.center_x + img_wid.center_x / 2.0 - self.img_width / 2.0,
                img_wid.center_y - self.img_height / 2.0),size=(500, 350))

    def show_edit_preview_color(self, c_wid, color_preview_data):
        with c_wid.canvas:
            c_wid.canvas.clear()
            for i in range(color_preview_data.shape[0]):
                Color(color_preview_data[i,0], color_preview_data[i,1], color_preview_data[i,2])
                Rectangle(pos=(i * c_wid.width / 33.0 + c_wid.x + c_wid.width / 130.0, c_wid.center_y ), size=(18, 18))

    def load_lut(self, wid, file_index, *largs):
        if file_index == 1:
            load_file = 1
        else:
            load_file = 2
        self.fileroot.set_file_index(load_file)
        self.fileroot.show_load()

    def save_lut(self, wid, *largs):
        self.fileroot.show_save()

    def change_axis(self, wid, *largs):
        self.axis = (self.axis + 1) % len(AxisList)
        # if (self.axis == self.slider_axis):
        #     self.slider_axis = (self.slider_axis + 1) % len(AxisList)
        self.update_label()
        self.show_lut_layer(wid)

    def show_3D_plot(self, wid, *largs):
        lut_layer_1 = None
        lut_layer_2 = None
        plot_type = 0
        if self.fileroot.lut_data1.is_empty() != True:
            lut_layer_1 = self.fileroot.lut_data1.get_lut_layer(self.axis, self.layer_index)
            plot_type |= 0b01
        if self.fileroot.lut_data2.is_empty() != True:
            lut_layer_2 = self.fileroot.lut_data2.get_lut_layer(self.axis, self.layer_index)
            plot_type |= 0b10
        for key, value in AxisPlotDic.iteritems():
            if key != AxisList[self.axis]:
                if plot_type == 0:
                    pass
                elif plot_type == 1:
                    self.plot3d.plot(lut_layer_1[:,:,value], None, key)
                elif plot_type == 2:
                    self.plot3d.plot(None, lut_layer_2[:,:,value], key)
                elif plot_type == 3:
                    self.plot3d.plot(lut_layer_1[:,:,value], lut_layer_2[:,:,value], key)
                # self.plot3d.plot(lut_layer[:,:,value], key)
            else:
                if plot_type == 0:
                    pass
                elif plot_type == 1:
                    self.plot3d.plot(lut_layer_1[:,:,value] + 1.0e-10, None, key)
                elif plot_type == 2:
                    self.plot3d.plot(None, lut_layer_2[:,:,value] + 1.0e-10, key)
                elif plot_type == 3:
                    self.plot3d.plot(lut_layer_1[:,:,value] + 1.0e-10, lut_layer_2[:,:,value] + 1.0e-10, key)

    def onlutlayerchange(self, wid, instance, value):
        # self.label.text = pattern.format(AxisList[self.axis], value)
        self.layer_index = int(value)
        self.update_label()
        self.show_lut_layer(wid)
        # self.update_edit_panel()

    def onslidervaluechange(self, c_wid, instance, value):
        self.slider_value = int(value)
        self.update_edit_panel(c_wid)

    def oneditcolorvalue(self, c_wid, instance, value):
        self.color_data_list[self.slider_list.index(instance)] = value
        if self.color_preview_data is not None:
            self.color_preview_data[:,self.edit_axis] = np.asarray(self.color_data_list)
            self.show_edit_preview_color(c_wid, self.color_preview_data)

        # print self.color_data_list
    def edit_press_callback(self, c_wid, instance):
        # print instance
        if self.slider_axis == self.axis:
             self.slider_axis = (self.slider_axis + 1) % len(AxisList)
        self.update_edit_panel(c_wid)

    def update_label(self):
        self.label.text = self.label_pattern.format(AxisList[self.axis], self.layer_index)

    def update_edit_label(self):
        self.edit_label.text = self.edit_label_pattern.format(AxisList[self.edit_axis],
        AxisList[self.axis], self.layer_index, AxisList[self.slider_axis], self.slider_value)


    def swap_axis(self, c_wid, *largs):
        self.slider_axis = (self.slider_axis + 1) % len(AxisList)
        self.color_preview_data = None
        if self.slider_axis == self.axis:
             self.slider_axis = (self.slider_axis + 1) % len(AxisList)
        self.update_edit_panel(c_wid)

    def change_editcolor(self, c_wid, *largs):
        self.edit_axis = (self.edit_axis + 1) % len(AxisList)
        self.update_edit_panel(c_wid)

    def apply_change(self, *largs):
        self.fileroot.lut_data1.set_edit_lutdata(self.axis, self.layer_index,
            self.edit_axis, self.slider_axis, self.slider_value, np.asarray(self.color_data_list))

    def show_curve(self, graph, function_input, *largs):
        plot = MeshLinePlot(color=[1, 0, 0, 1])
        x = np.linspace(0, 1, 100)
        nsp = NumericStringParser(x)
        y = nsp.eval(function_input.text)
        # y = np.power(x, 1/2.0)
        plot.points = [(p_x, p_y) for p_x, p_y in zip(x, y)]
        graph.add_plot(plot)

    def show_adjust_image(self, wid, function_input, *largs):
        im = PILImage.open("./img/wql_result.jpeg")
        im = im.transpose(PILImage.FLIP_TOP_BOTTOM)
        self.img_width, self.img_height = im.size
        self.img_array = np.array(im)
        print self.img_array
        temp_array = np.copy(self.img_array)
        texture = Texture.create(size=(500, 350), colorfmt="rgb")
        data = temp_array.tostring()
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")

        with wid.canvas:
            wid.canvas.clear()
            Rectangle(texture=texture, pos = (wid.center_x - self.img_width / 2.0,
                wid.center_y - self.img_height / 2.0),size=(500, 350))

    def preview_adjust(self, wid, function_input, function_input_red, function_input_green, function_input_blue, *largs):
        if function_input.text != "":
            x = self.img_array / 255.0
            nsp = NumericStringParser(x)
            preview_img_array = nsp.eval(function_input.text) * 255.0
        else:
            x = self.img_array / 255.0
            nsp = NumericStringParser(x[:,:,0],x[:,:,1],x[:,:,2])
            if function_input_red.text != "":
                x[:,:,0] = nsp.eval(function_input_red.text)
            if function_input_green.text != "":
                x[:,:,1] = nsp.eval(function_input_green.text)
            if function_input_blue.text != "":
                x[:,:,2] = nsp.eval(function_input_blue.text)

            preview_img_array = x * 255.0

        texture = Texture.create(size=(500, 350), colorfmt="rgb")
        preview_img_array = np.uint8(preview_img_array)
        # print preview_img_array
        data = preview_img_array.tostring()
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")

        with wid.canvas:
            wid.canvas.clear()
            Rectangle(texture=texture, pos = (wid.center_x - self.img_width / 2.0,
                wid.center_y - self.img_height / 2.0),size=(500, 350))

        # else:
        #     x = self.fileroot.lut_data1.get_lut_by_channel(0)
        #     y = self.fileroot.lut_data1.get_lut_by_channel(1)
        #     z = self.fileroot.lut_data1.get_lut_by_channel(2)
        #     nsp = NumericStringParser(x, y, z)
        #     new_lut = nsp.eval(function_input.text)
        #     self.fileroot.lut_data1.set_lut_by_channel(AxisPlotDic[self.toggle_type], new_lut)

    def apply_adjust(self, function_input, function_input_red, function_input_green, function_input_blue, *largs):
        if function_input.text != "":
            x = self.fileroot.lut_data1.get_lut()
            nsp = NumericStringParser(x)
            new_lut = nsp.eval(function_input.text)
            self.fileroot.lut_data1.set_lut(new_lut)
        else:
            x = self.fileroot.lut_data1.get_lut_by_channel(0)
            y = self.fileroot.lut_data1.get_lut_by_channel(1)
            z = self.fileroot.lut_data1.get_lut_by_channel(2)
            nsp = NumericStringParser(x, y, z)
            if function_input_red.text != "":
                new_lut = nsp.eval(function_input_red.text)
                self.fileroot.lut_data1.set_lut_by_channel(0, new_lut)

            if function_input_green.text != "":
                new_lut = nsp.eval(function_input_green.text)
                self.fileroot.lut_data1.set_lut_by_channel(1, new_lut)

            if function_input_blue.text != "":
                new_lut = nsp.eval(function_input_blue.text)
                self.fileroot.lut_data1.set_lut_by_channel(2, new_lut)

    def function_input_clear(self, instance):
        print instance

    def build(self):
        tp = TabbedPanel()

        wid = Widget(size_hint=(0.9, 1))
        slider = Slider(min=0, max=32, value=0, value_track=True, orientation='vertical',
            step=1.0, value_track_color=[1, 0, 0, 1], size_hint=(0.1, 1))
        self.label_pattern = "Axis {} : {}"
        self.edit_label_pattern = "Edit Color {}. Keep Axis {} : Layer Index {}. Slide Axis {} : {}"
        self.color_preview_data = None
        self.fileroot = Root()
        self.axis = 0
        self.edit_axis = 2
        self.slider_axis = 1
        self.slider_value = 0
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

        btn_save_lut = Button(text='Save LUT',
                           on_press=partial(self.save_lut, wid))

        layout = BoxLayout(size_hint=(1, None), height=50)
        layout.add_widget(btn_load_lut1)
        layout.add_widget(btn_load_lut2)
        layout.add_widget(btn_showlayer)
        layout.add_widget(btn_showplot)
        layout.add_widget(btn_double)
        layout.add_widget(btn_save_lut)
        layout.add_widget(self.label)

        root = BoxLayout(orientation='vertical')
        root.add_widget(upper_layout)
        root.add_widget(layout)
        slider.bind(value=partial(self.onlutlayerchange, wid))
        tp.default_tab_text = "Analysis"
        tp.background_color = (0,0,0,1)
        tp.default_tab_content = root

        #Edit tab define
        th_edittab_head = TabbedPanelHeader(text='Edit')

        slider_layout = BoxLayout(size_hint=(1, 0.9))
        color_wid = Widget(size_hint=(1, 0.1))
        self.slider_list = []
        for i in range(33):
            self.slider_list.append(Slider(min=0, max=1, value=0, value_track=False, orientation='vertical',
            cursor_size=(18,18), step = 0.000001, background_width = 0))
        for slider_item in self.slider_list:
            slider_layout.add_widget(slider_item)
            slider_item.bind(value=partial(self.oneditcolorvalue,color_wid))

        edit_preview = BoxLayout(orientation='vertical',size_hint=(0.9, 1))
        edit_preview.add_widget(slider_layout)
        edit_preview.add_widget(color_wid)

        edit_layout_upper = BoxLayout()
        c_slider = Slider(min=0, max=32, value=0, value_track=True, orientation='vertical',
        step=1.0, value_track_color=[1, 0, 0, 1], size_hint=(0.1, 1))
        c_slider.bind(value=partial(self.onslidervaluechange, color_wid))
        edit_layout_upper.add_widget(edit_preview)
        edit_layout_upper.add_widget(c_slider)

        edit_layout_lower = BoxLayout(size_hint=(1, None), height=50)
        btn_swap_axis = Button(text='Swap Slider Axis',size_hint=(0.12, 1),
                            on_press=partial(self.swap_axis, color_wid))

        btn_change_editcolor = Button(text='Change Edit Color',size_hint=(0.12, 1),
                            on_press=partial(self.change_editcolor, color_wid))

        btn_apply_change = Button(text='Apply Change',size_hint=(0.12, 1),
                            on_press=partial(self.apply_change))
        self.edit_label = Label(text=self.edit_label_pattern.format(AxisList[self.edit_axis],
        AxisList[self.axis], 0, AxisList[self.slider_axis], 0), size_hint=(0.5, 1))
        edit_layout_lower.add_widget(btn_swap_axis)
        edit_layout_lower.add_widget(btn_change_editcolor)
        edit_layout_lower.add_widget(btn_apply_change)
        edit_layout_lower.add_widget(self.edit_label)
        edit_layout = BoxLayout(orientation='vertical')
        edit_layout.add_widget(edit_layout_upper)
        edit_layout.add_widget(edit_layout_lower)

        th_edittab_head.content= edit_layout

        tp.add_widget(th_edittab_head)
        th_edittab_head.bind(on_press=partial(self.edit_press_callback, color_wid))

        th_adjusttab_head = TabbedPanelHeader(text='Adjust')
        function_label = Label(text='Function All: ', size_hint=(0.2, 1))
        function_label_red = Label(text='Function Red: ', size_hint=(0.2, 1))
        function_label_green = Label(text='Function Green: ', size_hint=(0.2, 1))
        function_label_blue = Label(text='Function Blue: ', size_hint=(0.2, 1))

        function_input = TextInput(text='', multiline=False, size_hint=(0.7, 1))
        function_input_red = TextInput(text='', multiline=False, size_hint=(0.7, 1))
        function_input_green = TextInput(text='', multiline=False, size_hint=(0.7, 1))
        function_input_blue = TextInput(text='', multiline=False, size_hint=(0.7, 1))

        btn_clear = Button(text='Clean',
                            on_press=partial(self.function_input_clear), size_hint=(0.1, 1))
        btn_clear_red = Button(text='Clean Red',
                            on_press=partial(self.function_input_clear), size_hint=(0.1, 1))
        btn_clear_green = Button(text='Clean Green',
                            on_press=partial(self.function_input_clear), size_hint=(0.1, 1))
        btn_clear_blue = Button(text='Clean Blue',
                            on_press=partial(self.function_input_clear), size_hint=(0.1, 1))

        function_layout = BoxLayout(size_hint=(1, None), height=30)
        function_layout.add_widget(function_label)
        function_layout.add_widget(function_input)
        function_layout.add_widget(btn_clear)

        function_layout_red = BoxLayout(size_hint=(1, None), height=30)
        function_layout_red.add_widget(function_label_red)
        function_layout_red.add_widget(function_input_red)
        function_layout_red.add_widget(btn_clear_red)

        function_layout_green = BoxLayout(size_hint=(1, None), height=30)
        function_layout_green.add_widget(function_label_green)
        function_layout_green.add_widget(function_input_green)
        function_layout_green.add_widget(btn_clear_green)

        function_layout_blue = BoxLayout(size_hint=(1, None), height=30)
        function_layout_blue.add_widget(function_label_blue)
        function_layout_blue.add_widget(function_input_blue)
        function_layout_blue.add_widget(btn_clear_blue)

        graph_layout = BoxLayout()

        graph = Graph(xlabel='In', ylabel='Out',
            x_ticks_major=1, y_ticks_major=1,
            y_grid_label=True, x_grid_label=True, padding=5,
            x_grid=True, y_grid=True, xmin=0, xmax=1, ymin=0, ymax=1, size_hint=(0.5, 1))
        # plot = MeshLinePlot(color=[1, 0, 0, 1])
        # plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]
        # graph.add_plot(plot)
        adjust_wid = Widget(size_hint=(0.5, 1))
        graph_layout.add_widget(graph)
        graph_layout.add_widget(adjust_wid)

        adjust_btn_layout = BoxLayout(size_hint=(1, None), height=50)
        btn_show_curve = Button(text='Show Curve',
                            on_press=partial(self.show_curve, graph, function_input))

        btn_adjust_image = Button(text='Show Image',
                            on_press=partial(self.show_adjust_image, adjust_wid, function_input))

        btn_preview_adjust = Button(text='Preview Adjust',
                            on_press=partial(self.preview_adjust, adjust_wid, function_input,
                            function_input_red, function_input_green, function_input_blue))

        btn_apply_adjust = Button(text='Apply Adjust',
                            on_press=partial(self.apply_adjust, function_input,
                            function_input_red, function_input_green, function_input_blue))
        adjust_btn_layout.add_widget(btn_show_curve)
        adjust_btn_layout.add_widget(btn_adjust_image)
        adjust_btn_layout.add_widget(btn_preview_adjust)
        adjust_btn_layout.add_widget(btn_apply_adjust)

        adjust_layout = BoxLayout(orientation='vertical')
        adjust_layout.add_widget(function_layout)
        adjust_layout.add_widget(function_layout_red)
        adjust_layout.add_widget(function_layout_green)
        adjust_layout.add_widget(function_layout_blue)
        adjust_layout.add_widget(graph_layout)
        adjust_layout.add_widget(adjust_btn_layout)
        th_adjusttab_head.content = adjust_layout
        tp.add_widget(th_adjusttab_head)

        #Preview tab define
        th_previewimg_head = TabbedPanelHeader(text='Preview')
        images_layout = BoxLayout(orientation='vertical')
        img_wid = Widget(size_hint=(1, 0.9))
        button_layout = BoxLayout(size_hint=(1, None), height=50)
        # img_array = cv2.imread('./img/wql_result.jpeg')

        btn_show_img = Button(text='Show Image',
                            on_press=partial(self.show_pre_image, img_wid))
        btn_apply_lut = Button(text='Apply LUT',
                            on_press=partial(self.onApplyLUTClick, img_wid))

        button_layout.add_widget(btn_show_img)
        button_layout.add_widget(btn_apply_lut)


        # th_previewimg_head.bind(on_press=partial(self.imgpreview_press_callback, img_wid, img_array))

        self.pb = ProgressBar(max=1000,size_hint=(1, 0.05))
        images_layout.add_widget(self.pb)
        images_layout.add_widget(img_wid)
        images_layout.add_widget(button_layout)
        th_previewimg_head.content = images_layout
        tp.add_widget(th_previewimg_head)

        return tp

Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

if __name__ == '__main__':
    LUTtoolsApp().run()
