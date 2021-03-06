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

import os
import csv
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

AxisList = ["Red", "Green", "Blue"]
# AxisList = ["Blue", "Green", "Red"] #Because of the reshape, the axis is different from lut doc
AxisPlotDic = {"Red" : 0, "Green" : 1, "Blue" : 2}
class Draw3DSurface(object):
    def __init__(self, step=0.030304):
        # self.lut_table = np.zeros((33, 33, 33, 3), dtype = np.float)
        self.X = np.arange(0, 1, step)
        self.Y = np.arange(0, 1, step)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
    def plot(self, Z1, Z2, plot_axis):
        if Z1 != None and Z2 != None:
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
        elif Z1 != None:
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
        elif Z2 != None:
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

    def get_lut_list(self):
        output_lut_table = self.lut_table.transpose((2,1,0,3))
        output_lut_table = np.copy(np.reshape(output_lut_table, (-1,3)))
        output_lut_table = output_lut_table.tolist()
        return output_lut_table

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

    def get_edit_lutdata(self, keep_axis, layer_index, edit_axis, slider_value):
        if keep_axis == 0:
            if edit_axis == 1:
                result = self.lut_table[layer_index, :, slider_value, edit_axis]
            else:
                result = self.lut_table[layer_index, slider_value, :, edit_axis]
        elif keep_axis == 1:
            if edit_axis == 0:
                result = self.lut_table[:, layer_index, slider_value, edit_axis]
            else:
                result = self.lut_table[slider_value, layer_index, :, edit_axis]
        elif keep_axis == 2:
            if edit_axis == 0:
                result = self.lut_table[:, slider_value, layer_index, edit_axis]
            else:
                result = self.lut_table[slider_value, :, layer_index, edit_axis]

        return np.copy(result)

    def get_edit_preview_colordata(self, keep_axis, layer_index, edit_axis,slider_value):
        if keep_axis == 0:
            if edit_axis == 1:
                result = self.lut_table[layer_index, :, slider_value, :]
            else:
                result = self.lut_table[layer_index, slider_value, :, :]
        elif keep_axis == 1:
            if edit_axis == 0:
                result = self.lut_table[:, layer_index, slider_value, :]
            else:
                result = self.lut_table[slider_value, layer_index, :, :]
        elif keep_axis == 2:
            if edit_axis == 0:
                result = self.lut_table[:, slider_value, layer_index, :]
            else:
                result = self.lut_table[slider_value, :, layer_index, :]

        return np.copy(result)

    def set_edit_lutdata(self, keep_axis, layer_index, edit_axis, slider_value, new_data):
        if keep_axis == 0:
            if edit_axis == 1:
                self.lut_table[layer_index, :, slider_value, edit_axis] = new_data
            else:
                self.lut_table[layer_index, slider_value, :, edit_axis] = new_data
        elif keep_axis == 1:
            if edit_axis == 0:
                self.lut_table[:, layer_index, slider_value, edit_axis] = new_data
            else:
                self.lut_table[slider_value, layer_index, :, edit_axis] = new_data
        elif keep_axis == 2:
            if edit_axis == 0:
                self.lut_table[:, slider_value, layer_index, edit_axis] = new_data
            else:
                self.lut_table[slider_value, :, layer_index, edit_axis] = new_data

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

    def saveLUT(self, path, filename, LUT_data):
        with open(os.path.join(path, filename), 'w') as lutfile:
            datawriter = csv.writer(lutfile, delimiter=' ')
            datawriter.writerow(["LUT_3D_SIZE","33"])
            datawriter.writerow(["LUT_3D_INPUT_RANGE", "0.0000000000", "1.0000000000"])
            datawriter.writerow([])
            output_list = LUT_data.get_lut_list()
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
        lut = self.readLUT(path, filename)
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

    def add_rects(self, wid, count, *largs):
        with wid.canvas:
            for x in range(count):
                Color(r(), 1, 1, mode='hsv')
                Rectangle(pos=(r() * wid.width + wid.x,
                               r() * wid.height + wid.y), size=(20, 20))

    def show_lut_layer(self, wid, *largs):
        with wid.canvas:
            wid.canvas.clear()
            # if self.fileroot.lut_data1.is_empty() == False:
            #     # self.update_edit_panel()
            #     lut_layer = self.fileroot.lut_data1.get_lut_layer(self.axis, self.layer_index)
            #     # print lut_layer.shape
            #     lut_layer = np.reshape(lut_layer, (-1,3))
            #     # print lut_layer.shape
            #     for i in range(lut_layer.shape[0]):
            #         x = i % 33
            #         y = i / 33
            #         Color(lut_layer[i,0], lut_layer[i,1], lut_layer[i,2])
            #         Rectangle(pos=(x * 25 + wid.x + 20,
            #                        y * 18 + wid.y + 20), size=(10, 10))
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
                # lut_layer = self.fileroot.lut_data2.get_lut_layer(self.axis, self.layer_index)
                # # print lut_layer.shape
                # lut_layer = np.reshape(lut_layer, (-1,3))
                # # print lut_layer.shape
                # for i in range(lut_layer.shape[0]):
                #     x = i % 33
                #     y = i / 33
                #     Color(lut_layer[i,0], lut_layer[i,1], lut_layer[i,2])
                #     Rectangle(pos=(x * 25 + wid.x + 32,
                #                    y * 18 + wid.y + 20), size=(10, 10))

    def update_edit_panel(self, c_wid):
        # lut_layer = self.fileroot.lut_data1.get_lut_layer(self.axis, self.layer_index)
        color_data = self.fileroot.lut_data1.get_edit_lutdata(self.axis, self.layer_index, self.edit_axis, self.slider_value)
        self.color_data_list = color_data.tolist()
        for sld, value in zip(self.slider_list, self.color_data_list):
            sld.value = value

        self.color_preview_data = self.fileroot.lut_data1.get_edit_preview_colordata(self.axis, self.layer_index,
                self.edit_axis, self.slider_value)
        self.show_edit_preview_color(c_wid, self.color_preview_data)
        self.update_edit_label()

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
        # if (self.axis == self.edit_axis):
        #     self.edit_axis = (self.edit_axis + 1) % len(AxisList)
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

    def reset_rects(self, wid, *largs):
        pass

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
        self.update_edit_panel(c_wid)

    def update_label(self):
        self.label.text = self.label_pattern.format(AxisList[self.axis], self.layer_index)

    def update_edit_label(self):
        self.edit_label.text = self.edit_label_pattern.format(AxisList[self.edit_axis],
        AxisList[self.axis], self.layer_index, self.slider_value)


    def swap_axis(self, c_wid, *largs):
        self.edit_axis = (self.edit_axis + 1) % len(AxisList)
        self.color_preview_data = None
        if self.edit_axis == self.axis:
             self.edit_axis = (self.edit_axis + 1) % len(AxisList)
        self.update_edit_panel(c_wid)

    def apply_change(self, *largs):
        self.fileroot.lut_data1.set_edit_lutdata(self.axis, self.layer_index,
            self.edit_axis, self.slider_value, np.asarray(self.color_data_list))

    def build(self):
        tp = TabbedPanel()

        wid = Widget(size_hint=(0.9, 1))
        slider = Slider(min=0, max=32, value=0, value_track=True, orientation='vertical',
        step=1.0, value_track_color=[1, 0, 0, 1], size_hint=(0.1, 1))
        self.label_pattern = "Axis {} : {}"
        self.edit_label_pattern = "Edit Color {}. Keep Axis {} : Layer Index {}. Slide : {}"
        self.color_preview_data = None
        self.fileroot = Root()
        self.axis = 0
        self.edit_axis = 1
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
        th_text_head = TabbedPanelHeader(text='Edit')

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
        btn_swap_axis = Button(text='Swap Axis',size_hint=(0.15, 1),
                            on_press=partial(self.swap_axis, color_wid))

        btn_apply_change = Button(text='Apply Change',size_hint=(0.15, 1),
                            on_press=partial(self.apply_change))
        self.edit_label = Label(text=self.edit_label_pattern.format(AxisList[self.edit_axis],
        AxisList[self.axis], 0, 0), size_hint=(0.5, 1))
        edit_layout_lower.add_widget(btn_swap_axis)
        edit_layout_lower.add_widget(btn_apply_change)
        edit_layout_lower.add_widget(self.edit_label)
        edit_layout = BoxLayout(orientation='vertical')
        edit_layout.add_widget(edit_layout_upper)
        edit_layout.add_widget(edit_layout_lower)

        th_text_head.content= edit_layout

        tp.add_widget(th_text_head)
        th_text_head.bind(on_press=partial(self.edit_press_callback, color_wid))

        return tp

Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

if __name__ == '__main__':
    LUTtoolsApp().run()
