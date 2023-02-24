
# Development notes for glu-puff-analysis

## Table of Contents
- [Clipboard (Frequently copied text)](#clipboard-frequently-copied-text)
- [Future Features / minor changes](#future-features--minor-changes)
  - [1. (user) Add option to save graph as .svg or .pdf:](#1-user-add-option-to-save-graph-as-svg-or-pdf)
  - [2. (user) Save ROI's and other preprocessed data within project folder](#2-user-save-rois-and-other-preprocessed-data-within-project-folder)
  - [3. (dev env) Add GUI on WSL (to allow napari in WSL therefore testing in WSL as substitute for mac)](#3-dev-env-add-gui-on-wsl-to-allow-napari-in-wsl-therefore-testing-in-wsl-as-substitute-for-mac)
  - [4. (user) Preset Napari parameters and disable unimplemented ones](#4-user-preset-napari-parameters-and-disable-unimplemented-ones)
  - [5. (user) Add loading bar when plotting files (can take a while and no visual indicator of progress)](#5-user-add-loading-bar-when-plotting-files-can-take-a-while-and-no-visual-indicator-of-progress)
  - [6. (user) Open folder with plots after script completes](#6-user-open-folder-with-plots-after-script-completes)
  - [7. (user) Add conversion from .oif to .tif files to start of script](#7-user-add-conversion-from-oif-to-tif-files-to-start-of-script)
  - [8. (user) Automatically or manually identify the needle](#8-user-automatically-or-manually-identify-the-needle)
  - [9. (user) Order ROI's based on distance instead of order drawn](#9-user-order-rois-based-on-distance-instead-of-order-drawn)
- [Bugs](#bugs)
  - [1. (user bug)](#1-user-bug)
  - [2. (dev env bug)](#2-dev-env-bug)
  - [3. (user bug)](#3-user-bug)
  - [3. (user bug)](#3-user-bug-1)
  - [4. (user bug)](#4-user-bug)
  - [5. (user bug)](#5-user-bug)
  - [6. (user bug)](#6-user-bug)
  - [7. (user bug)](#7-user-bug)
- [Error Messages](#error-messages)
  - [1. ERROR MESSAGE (bug 1):](#1-error-message-bug-1)
  - [2. RUNTIME ERROR (bug 3):](#2-runtime-error-bug-3)
	

## Clipboard (Frequently copied text)

```
/mnt/c/Users/ahshenas/Lab/mockglupuffdata/20221208_6s-ss96-glutpuff_01.tif
```


## Future Features / minor changes 

### 1. (user) Add option to save graph as .svg or .pdf:
	ex: python GluPuff_Pipeline.py -f .svg
	alternatively, prompt user for one or multiple file types to save as after roi's are selected
	
### 2. (user) Save ROI's and other preprocessed data within project folder 
	(if sizes are reasonable, if not reduce sizes or just save ROI's to speed up ROI drawing)
	Example flow:
		User selects file(s)
		If Preprocessed data is found, User is asked if they want to plot it (with options for different file formats)
		
### 3. (dev env) Add GUI on WSL (to allow napari in WSL therefore testing in WSL as substitute for mac)
	https://scottspence.com/posts/gui-with-wsl
	https://nickymeuleman.netlify.app/blog/gui-on-wsl2-cypress

### 4. (user) Preset Napari parameters and disable unimplemented ones
	Set colormap to viridis (better visibility against ROI outlines)
	Lower image gamma to increase brightness
	Lower opacity of shapes to see image better
	disable shape tools that are not implemented (circular and rectangular selection, etc)

### 5. (user) Add loading bar when plotting files (can take a while and no visual indicator of progress)

### 6. (user) Open folder with plots after script completes

### 7. (user) Add conversion from .oif to .tif files to start of script

### 8. (user) Automatically or manually identify the needle
	Automatic identification using directional filtering?
	
### 9. (user) Order ROI's based on distance instead of order drawn
	Find the greatest distance between centers
	label ROI's based on that order


## Bugs

### 1. (user bug)
Description: "Unable to move the cache: Access is denied"<br>
Notes: Full error message in error message section [Err 1](#1-error-message-bug-1)

### 2. (dev env bug)
Description: High memorry usage by vmmem<br>
Note: Possible solutions -
	https://github.com/microsoft/WSL/issues/4166#issuecomment-526725261
	https://learn.microsoft.com/en-us/windows/wsl/release-notes#build-18945

### 3. (user bug)
Description: Runtime AssertionError
Notes: Happened 7 images into ROI drawing
	Could be caused by drawing a polygon w less than 3 points?
	Full error message in error message section (Err 2)

### 3. (user bug)
Description: Napari "RuntimeWarning: invalid value encountered in cast!"
Notes: Occurs when Drawing ROI's

### 4. (user bug)
Description: On MAC- Fails after tiffs selected with permission error when closing files

### 5. (user bug)
Description: Fix number format for colorbars
	Can become long in scientific format and push label into ROI plot
	
### 6. (user bug)
Description: .sh script broken

### 7. (user bug)
Description: If too few ROI's are selected the image will have a large aspect ratio (very short compared to number of frames)
Notes: Possible solution - Increase image aspect greatly with few reigons
	ex: 1 region -> 64 aspect when plotting heatmap
		2 regions -> 32 aspect
	Also needs to consider length of trials

## Error Messages

### 1. ERROR MESSAGE (bug 1):
    ```
    (glupuff) C:\Users\ahshenas\Documents\GitHub\glu-puff-analysis>GluPuff_Pipeline.py

    (glupuff) C:\Users\ahshenas\Documents\GitHub\glu-puff-analysis>
    [20064:0213/161350.154:ERROR:cache_util_win.cc(20)] Unable to move the cache: Access is denied. (0x5)
    [20064:0213/161350.154:ERROR:cache_util.cc(145)] Unable to move cache folder C:\Users\ahshenas\AppData\Roaming\Code\Cache\Cache_Data to C:\Users\ahshenas\AppData\Roaming\Code\Cache\old_Cache_Data_000
    [20064:0213/161350.168:ERROR:disk_cache.cc(196)] Unable to create cache
    ```


### 2. RUNTIME ERROR (bug 3):
    ```
    draw ROI's for image 7
    C:\Users\ahshenas\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\base\base.py:1632: RuntimeWarning: invalid value encountered in cast!
    C:\Users\ahshenas\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\base\base.py:1632: RuntimeWarning: invalid value encountered in cast!
    ---------------------------------------------------------------------------
    AssertionError                            Traceback (most recent call last)
    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\app\backends\_qt.py:532, in QtBaseCanvasBackend.mouseMoveEvent(self=<vispy.app.backends._qt.CanvasBackendDesktop object>, ev=<PyQt5.QtGui.QMouseEvent object>)
        530 if self._vispy_canvas is None:
        531     return
    --> 532 self._vispy_mouse_move(
            self = <vispy.app.backends._qt.CanvasBackendDesktop object at 0x0000022705AE9EE0>
            ev = <PyQt5.QtGui.QMouseEvent object at 0x00000226E9001B80>
        533     native=ev,
        534     pos=_get_event_xy(ev),
        535     modifiers=self._modifiers(ev),
        536 )

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\app\base.py:216, in BaseCanvasBackend._vispy_mouse_move(self=<vispy.app.backends._qt.CanvasBackendDesktop object>, **kwargs={'buttons': [], 'last_event': <MouseEvent blocked=False button=1 buttons=[] de...ces=[] time=1676336225.314424 type=mouse_release>, 'last_mouse_press': None, 'modifiers': (), 'native': <PyQt5.QtGui.QMouseEvent object>, 'pos': (1145, 1031), 'press_event': None})
        213 else:
        214     kwargs['button'] = self._vispy_mouse_data['press_event'].button
    --> 216 ev = self._vispy_canvas.events.mouse_move(**kwargs)
            self = <vispy.app.backends._qt.CanvasBackendDesktop object at 0x0000022705AE9EE0>
            self._vispy_canvas.events.mouse_move = <vispy.util.event.EventEmitter object at 0x00000226E59E5AF0>
            kwargs = {'native': <PyQt5.QtGui.QMouseEvent object at 0x00000226E9001B80>, 'pos': (1145, 1031), 'modifiers': (), 'buttons': [], 'press_event': None, 'last_event': <MouseEvent blocked=False button=1 buttons=[] delta=[0. 0.] handled=False is_dragging=True last_event=None modifiers=() native=<PyQt5.QtGui.QMouseEvent object at 0x00000226E9001B80> pos=[1143 1029] press_event=MouseEvent source=None sources=[] time=1676336225.314424 type=mouse_release>, 'last_mouse_press': None}
            self._vispy_canvas.events = <vispy.util.event.EmitterGroup object at 0x00000226E59EF280>
            self._vispy_canvas = <VispyCanvas (PyQt5) at 0x226e59fb550>
        217 self._vispy_mouse_data['last_event'] = ev
        218 return ev

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\util\event.py:453, in EventEmitter.__call__(self=<vispy.util.event.EventEmitter object>, *args=(), **kwargs={'buttons': [], 'last_event': <MouseEvent blocked=False button=1 buttons=[] de...ces=[] time=1676336225.314424 type=mouse_release>, 'last_mouse_press': None, 'modifiers': (), 'native': <PyQt5.QtGui.QMouseEvent object>, 'pos': (1145, 1031), 'press_event': None})
        450 if self._emitting > 1:
        451     raise RuntimeError('EventEmitter loop detected!')
    --> 453 self._invoke_callback(cb, event)
            event = <MouseEvent blocked=False button=None buttons=[] delta=[0. 0.] handled=False is_dragging=False last_event=MouseEvent modifiers=() native=<PyQt5.QtGui.QMouseEvent object at 0x00000226E9001B80> pos=[1145 1031] press_event=None source=None sources=[] time=1676336225.3462903 type=mouse_move>
            self = <vispy.util.event.EventEmitter object at 0x00000226E59E5AF0>
            cb = <bound method QtViewer.on_mouse_move of <napari._qt.qt_viewer.QtViewer object at 0x0000022701441C10>>
        454 if event.blocked:
        455     break

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\util\event.py:471, in EventEmitter._invoke_callback(self=<vispy.util.event.EventEmitter object>, cb=<bound method QtViewer.on_mouse_move of <napari._qt.qt_viewer.QtViewer object>>, event=<MouseEvent blocked=False button=None buttons=[]...urces=[] time=1676336225.3462903 type=mouse_move>)
        469     cb(event)
        470 except Exception:
    --> 471     _handle_exception(self.ignore_callback_errors,
            self = <vispy.util.event.EventEmitter object at 0x00000226E59E5AF0>
            cb = <bound method QtViewer.on_mouse_move of <napari._qt.qt_viewer.QtViewer object at 0x0000022701441C10>>
            event = <MouseEvent blocked=False button=None buttons=[] delta=[0. 0.] handled=False is_dragging=False last_event=MouseEvent modifiers=() native=<PyQt5.QtGui.QMouseEvent object at 0x00000226E9001B80> pos=[1145 1031] press_event=None source=None sources=[] time=1676336225.3462903 type=mouse_move>
            (cb, event) = (<bound method QtViewer.on_mouse_move of <napari._qt.qt_viewer.QtViewer object at 0x0000022701441C10>>, <MouseEvent blocked=False button=None buttons=[] delta=[0. 0.] handled=False is_dragging=False last_event=MouseEvent modifiers=() native=<PyQt5.QtGui.QMouseEvent object at 0x00000226E9001B80> pos=[1145 1031] press_event=None source=None sources=[] time=1676336225.3462903 type=mouse_move>)
        472                       self.print_callback_errors,
        473                       self, cb_event=(cb, event))

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\util\event.py:469, in EventEmitter._invoke_callback(self=<vispy.util.event.EventEmitter object>, cb=<bound method QtViewer.on_mouse_move of <napari._qt.qt_viewer.QtViewer object>>, event=<MouseEvent blocked=False button=None buttons=[]...urces=[] time=1676336225.3462903 type=mouse_move>)
        467 def _invoke_callback(self, cb, event):
        468     try:
    --> 469         cb(event)
            cb = <bound method QtViewer.on_mouse_move of <napari._qt.qt_viewer.QtViewer object at 0x0000022701441C10>>
            event = <MouseEvent blocked=False button=None buttons=[] delta=[0. 0.] handled=False is_dragging=False last_event=MouseEvent modifiers=() native=<PyQt5.QtGui.QMouseEvent object at 0x00000226E9001B80> pos=[1145 1031] press_event=None source=None sources=[] time=1676336225.3462903 type=mouse_move>
        470     except Exception:
        471         _handle_exception(self.ignore_callback_errors,
        472                           self.print_callback_errors,
        473                           self, cb_event=(cb, event))

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\_qt\qt_viewer.py:1077, in QtViewer.on_mouse_move(self=<napari._qt.qt_viewer.QtViewer object>, event=<MouseEvent blocked=False button=None buttons=[]...urces=[] time=1676336225.3462903 type=mouse_move>)
    1069 def on_mouse_move(self, event):
    1070     """Called whenever mouse moves over canvas.
    1071
    1072     Parameters
    (...)
    1075         The vispy event that triggered this method.
    1076     """
    -> 1077     self._process_mouse_event(mouse_move_callbacks, event)
            event = <MouseEvent blocked=False button=None buttons=[] delta=[0. 0.] handled=False is_dragging=False last_event=MouseEvent modifiers=() native=<PyQt5.QtGui.QMouseEvent object at 0x00000226E9001B80> pos=[1145 1031] press_event=None source=None sources=[] time=1676336225.3462903 type=mouse_move>
            self = <napari._qt.qt_viewer.QtViewer object at 0x0000022701441C10>

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\_qt\qt_viewer.py:1026, in QtViewer._process_mouse_event(self=<napari._qt.qt_viewer.QtViewer object>, mouse_callbacks=<function mouse_move_callbacks>, event=<ReadOnlyWrapper at 0x00000226E6C82AC0 for MouseEvent>)
    1024 layer = self.viewer.layers.selection.active
    1025 if layer is not None:
    -> 1026     mouse_callbacks(layer, event)
            event = <ReadOnlyWrapper at 0x00000226E6C82AC0 for MouseEvent at 0x00000226E5A12550>
            layer = <Shapes layer 'Shapes' at 0x226e621fc70>
            mouse_callbacks = <function mouse_move_callbacks at 0x00000226E184C9D0>

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\utils\interactions.py:164, in mouse_move_callbacks(obj=<Shapes layer 'Shapes'>, event=<ReadOnlyWrapper at 0x00000226E6C82AC0 for MouseEvent>)
        161 if not event.is_dragging:
        162     # if not dragging simply call the mouse move callbacks
        163     for mouse_move_func in obj.mouse_move_callbacks:
    --> 164         mouse_move_func(obj, event)
            mouse_move_func = <function add_path_polygon_creating at 0x00000226E33BB5E0>
            event = <ReadOnlyWrapper at 0x00000226E6C82AC0 for MouseEvent at 0x00000226E5A12550>
            obj = <Shapes layer 'Shapes' at 0x226e621fc70>
        166 # for each drag callback get the current generator
        167 for func, gen in tuple(obj._mouse_drag_gen.items()):
        168     # save the event current event

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\shapes\_shapes_mouse_bindings.py:216, in add_path_polygon_creating(layer=<Shapes layer 'Shapes'>, event=<ReadOnlyWrapper at 0x00000226E6C82AC0 for MouseEvent>)
        214 if layer._is_creating:
        215     coordinates = layer.world_to_data(event.position)
    --> 216     _move(layer, coordinates)
            layer = <Shapes layer 'Shapes' at 0x226e621fc70>
            coordinates = (0.0, 0.0, 206.9708395463223, 63.91553762899167)

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\shapes\_shapes_mouse_bindings.py:528, in _move(layer=<Shapes layer 'Shapes'>, coordinates=(0.0, 0.0, 206.9708395463223, 63.91553762899167))
        526 vertices = layer._data_view.shapes[index].data
        527 vertices[vertex] = coordinates
    --> 528 layer._data_view.edit(index, vertices, new_type=new_type)
            layer = <Shapes layer 'Shapes' at 0x226e621fc70>
            index = 1
            vertices = <class 'numpy.ndarray'> (3, 4) float64
            layer._data_view = <napari.layers.shapes._shape_list.ShapeList object at 0x00000226E67FC5E0>
            new_type = None
        529 shapes = layer.selected_data
        530 layer._selected_box = layer.interaction_box(shapes)

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\shapes\_shape_list.py:737, in ShapeList.edit(self=<napari.layers.shapes._shape_list.ShapeList object>, index=1, data=<class 'numpy.ndarray'> (3, 4) float64, face_color=None, edge_color=None, new_type=None)
        735 else:
        736     shape = self.shapes[index]
    --> 737     shape.data = data
            shape = <napari.layers.shapes._shapes_models.polygon.Polygon object at 0x00000226E5404070>
            data = <class 'numpy.ndarray'> (3, 4) float64
        739 if face_color is not None:
        740     self._face_color[index] = face_color

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\shapes\_shapes_models\_polgyon_base.py:76, in PolygonBase.data(self=<napari.layers.shapes._shapes_models.polygon.Polygon object>, data=<class 'numpy.ndarray'> (3, 4) float64)
        67     raise ValueError(
        68         trans._(
        69             "Shape needs at least two vertices, {number} provided.",
    (...)
        72         )
        73     )
        75 self._data = data
    ---> 76 self._update_displayed_data()
            self = <napari.layers.shapes._shapes_models.polygon.Polygon object at 0x00000226E5404070>

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\shapes\_shapes_models\_polgyon_base.py:81, in PolygonBase._update_displayed_data(self=<napari.layers.shapes._shapes_models.polygon.Polygon object>)
        79 """Update the data that is to be displayed."""
        80 # For path connect every all data
    ---> 81 self._set_meshes(
            self = <napari.layers.shapes._shapes_models.polygon.Polygon object at 0x00000226E5404070>
            self._filled = True
            self._closed = True
        82     self.data_displayed, face=self._filled, closed=self._closed
        83 )
        84 self._box = create_box(self.data_displayed)
        86 data_not_displayed = self.data[:, self.dims_not_displayed]

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\shapes\_shapes_models\shape.py:223, in Shape._set_meshes(self=<napari.layers.shapes._shapes_models.polygon.Polygon object>, data=<class 'numpy.ndarray'> (3, 2) float64, closed=True, face=True, edge=True)
        221 if not is_collinear(clean_data[:, -2:]):
        222     if clean_data.shape[1] == 2:
    --> 223         vertices, triangles = triangulate_face(clean_data)
            triangles = <class 'numpy.ndarray'> (6, 3) int32
            clean_data = <class 'numpy.ndarray'> (3, 2) float64
        224     elif len(np.unique(clean_data[:, 0])) == 1:
        225         val = np.unique(clean_data[:, 0])

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\napari\layers\shapes\_shapes_utils.py:562, in triangulate_face(data=<class 'numpy.ndarray'> (3, 2) float64)
        560     vertices, triangles = res['vertices'], res['triangles']
        561 else:
    --> 562     vertices, triangles = PolygonData(vertices=data).triangulate()
            data = <class 'numpy.ndarray'> (3, 2) float64
        564 triangles = triangles.astype(np.uint32)
        566 return vertices, triangles

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\geometry\polygon.py:125, in PolygonData.triangulate(self=<vispy.geometry.polygon.PolygonData object>)
        122     edges[:, 1] = edges[:, 0] + 1
        124 tri = Triangulation(self._vertices, edges)
    --> 125 tri.triangulate()
            tri = <vispy.geometry.triangulation.Triangulation object at 0x0000022702DEC9D0>
        126 return tri.pts, tri.tris

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\geometry\triangulation.py:186, in Triangulation.triangulate(self=<vispy.geometry.triangulation.Triangulation object>)
        182     if i in self._tops:
        183         for j in self._bottoms[self._tops == i]:
        184             # Make sure edge (j, i) is present in mesh
        185             # because edge event may have created a new front list
    --> 186             self._edge_event(i, j)
            i = 4
            self = <vispy.geometry.triangulation.Triangulation object at 0x0000022702DEC9D0>
            j = 2
        187             front = self._front
        189 self._finalize()

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\geometry\triangulation.py:405, in Triangulation._edge_event(self=<vispy.geometry.triangulation.Triangulation object>, i=4, j=2)
        403 while len(polygon) > 2:
        404     ind = np.argmax(dist)
    --> 405     self._add_tri(polygon[ind], polygon[ind-1],
            polygon = [4, 3, 2]
            self = <vispy.geometry.triangulation.Triangulation object at 0x0000022702DEC9D0>
            ind = 0
        406                   polygon[ind+1])
        407     polygon.pop(ind)
        408     dist.pop(ind)

    File ~\Anaconda3\envs\glupufftest\lib\site-packages\vispy\geometry\triangulation.py:772, in Triangulation._add_tri(self=<vispy.geometry.triangulation.Triangulation object>, a=4, b=2, c=3)
        770 else:
        771     assert (b, a) not in self._edges_lookup
    --> 772     assert (c, b) not in self._edges_lookup
            b = 2
            c = 3
            self = <vispy.geometry.triangulation.Triangulation object at 0x0000022702DEC9D0>
            self._edges_lookup = {(2, 1): 3, (1, 3): 2, (3, 2): 1, (3, 1): 4, (1, 4): 3, (4, 3): 1}
            (c, b) = (3, 2)
        773     assert (a, c) not in self._edges_lookup
        774     self._edges_lookup[(b, a)] = c

    AssertionError:
    draw ROI's for image 8
    ```