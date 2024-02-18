import inkex
import math
from math import cos, sin
import numpy as np
import random


delim_list = {"tab": "\t", "space": " ", "lf": "\\n", "crlf": "\r\n"}


# Vector2dをnumpy配列に変換
def v2d_to_np(v2d: inkex.Vector2d):
    return np.array([v2d.x, v2d.y])


# numpy配列をVector2dに変換
def np_to_v2d(v: np.ndarray):
    return inkex.Vector2d(v[0], v[1])


# numpy配列を回転
def rotate(v: np.ndarray, angle: float):
    rad = np.radians(angle)
    rot = np.array([[cos(rad), -sin(rad)], [sin(rad), cos(rad)]])
    return np.dot(rot, v)


# numpy配列の大きさを1に正規化
def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)


# 有効な要素が2つ以上あるか
# グループ化するかどうかの判定に使う
def has_multiple_valid_elements(*args):
    count = 0
    for arg in args:
        if arg:
            count += 1
            if count >= 2:
                return True
    return False


# idを生成
# セットしようとしたidが重複しているとランダムなidにされてしまうので、その対策
def make_id(id_str: str):
    return f"{id_str}_{str(random.randint(0, 9999)).zfill(4)}"


class TextElement(inkex.TextElement):
    def __init__(self):
        super().__init__()

    # y座標をいい感じにする
    def set_y(self, y: float):
        self.set("y", str(y + float(self.style["font-size"]) * 0.3))

    def set_position(self, x: float, y: float):
        self.set("x", str(x))
        self.set_y(y)


class Axis:
    def __init__(
        self, min: float, max: float, start: inkex.Vector2d, end: inkex.Vector2d
    ):
        if min < max:
            self.min = min
            self.max = max
        elif min > max:
            self.min = max
            self.max = min
        else:
            raise inkex.AbortExtension(
                "エラー：軸の最小値と最大値が等しいです。異なる値を指定してください。"
            )
        self.start = start
        self.end = end
        self.vector = np.array([end.x - start.x, end.y - start.y])
        if np.linalg.norm(self.vector) == 0:
            raise Exception("軸の始点と終点が同じです。")

    # 目盛り、数字を配置する値を求める
    def get_tick_values(self, step: float, offset: float):
        if step <= 0:
            return []

        # tick_value = step * i + offset
        # min以上の最小のtick_valueとなるi
        min_tick_i = math.ceil((self.min - offset) / step)
        # max以下の最大のtick_valueとなるi
        max_tick_i = math.floor((self.max - offset) / step)
        values = []
        for i in range(min_tick_i, max_tick_i + 1):
            values.append(step * i + offset)
        return values

    # 目盛りを生成
    def get_ticks(
        self,
        step: float,
        offset: float,
        size: float,
        stroke_width: float,
        direction: int = 1,
    ):
        tick_values = self.get_tick_values(step, offset)
        ticks = inkex.Group()
        style = GraphPlotter.base_stroke_style.copy()
        style["stroke-width"] = str(stroke_width)
        for tick_value in tick_values:
            # 目盛り線の始点位置ベクトル
            tick_start_v = self.get_position(tick_value)
            # 目盛り線のベクトル
            np_tick_v = size * normalize(rotate(self.vector, 90 * direction))
            tick_v = np_to_v2d(np_tick_v)

            tick = inkex.Line(
                x1=str(tick_start_v.x),
                y1=str(tick_start_v.y),
                x2=str(tick_start_v.x + tick_v.x),
                y2=str(tick_start_v.y + tick_v.y),
            )
            tick.style = style
            ticks.add(tick)
        return ticks

    # 数値を生成
    def get_numbers(self, step: float, offset: float):
        num_values = self.get_tick_values(step, offset)
        numbers = inkex.Group()
        for num_value in num_values:
            # 数値の位置ベクトル
            pos = self.get_position(num_value)
            number_el = TextElement()
            self.set_numbers_style_and_position(number_el, pos)
            number_el.text = str(num_value)
            numbers.add(number_el)
        return numbers

    def set_numbers_style_and_position(self, text, pos):
        raise NotImplementedError(
            "サブクラスのオーバライドされたメソッドを呼び出してください。"
        )

    # 値から座標を求める
    def get_position(self, value: float):
        start_pos = v2d_to_np(self.start)
        range = self.max - self.min
        position = start_pos + value / range * self.vector
        return np_to_v2d(position)


class XAxis(Axis):
    placements = {"top": 1, "bottom": -1}

    def __init__(
        self,
        min: float,
        max: float,
        start_x: float,
        end_x: float,
        y: float,
        plasement: str = "bottom",
    ):
        start = inkex.Vector2d(start_x, y)
        end = inkex.Vector2d(end_x, y)
        super().__init__(min, max, start, end)
        self.placement = self.placements[plasement]

    # 目盛りを生成
    def get_ticks(self, step: float, offset: float, size: float, stroke_width: float):
        reverse_dir = (
            1 if self.vector[0] > 0 else -1
        )  # x軸の向きによって目盛りの向きを変える
        direction = self.placement * reverse_dir
        return super().get_ticks(step, offset, size, stroke_width, direction)

    def set_numbers_style_and_position(self, text_el, pos):
        style = GraphPlotter.base_text_style.copy()
        style["text-anchor"] = "middle"
        style["text-align"] = "center"
        text_el.style = style
        if self.placement == self.placements["top"]:
            text_el.set_position(pos.x, pos.y - float(style["font-size"]) * 0.6)
        else:
            text_el.set_position(pos.x, pos.y + float(style["font-size"]) * 0.9)

    # ラベルを生成
    def get_label(self, text: str, position: float):
        label = TextElement()
        label.style = GraphPlotter.base_text_style.copy()
        cx = self.start.x + self.vector[0] / 2
        label.set_position(
            x=self.start.x + (self.end.x - self.start.x) / 2,
            y=self.start.y - self.placement * position,
        )
        # 90度回転
        # label.set("transform", "rotate(90 " + str(label.get("x")) + " " + str(label.get("y")) + ")")
        label.text = text
        return label

    # 値からx座標を求める
    def get_x(self, value: float):
        return self.get_position(value).x


class YAxis(Axis):
    placements = {"left": 1, "right": -1}

    def __init__(
        self,
        min: float,
        max: float,
        start_y: float,
        end_y: float,
        x: float,
        plasement: str = "left",
    ):
        start = inkex.Vector2d(x, start_y)
        end = inkex.Vector2d(x, end_y)
        super().__init__(min, max, start, end)
        self.placement = self.placements[plasement]

    # 目盛りを生成
    def get_ticks(self, step: float, offset: float, size: float, stroke_width: float):
        reverse_dir = (
            1 if self.vector[1] < 0 else -1
        )  # y軸の向きによって目盛りの向きを変える
        direction = self.placement * reverse_dir
        return super().get_ticks(step, offset, size, stroke_width, direction)

    def set_numbers_style_and_position(self, text_el, pos):
        style = GraphPlotter.base_text_style.copy()
        if self.placement == self.placements["left"]:
            style["text-anchor"] = "end"
            style["text-align"] = "right"
            text_el.style = style
            text_el.set_position(pos.x - float(style["font-size"]) * 0.4, pos.y)
        else:
            style["text-anchor"] = "start"
            style["text-align"] = "left"
            text_el.style = style
            text_el.set_position(pos.x + float(style["font-size"]) * 0.4, pos.y)

    # ラベルを生成
    def get_label(self, text: str, position: float):
        label = TextElement()
        label.style = GraphPlotter.base_text_style.copy()
        label.set_position(
            x=self.start.x - self.placement * position,
            y=self.start.y + (self.end.y - self.start.y) / 2,
        )
        label.set(
            "transform",
            "rotate(-90 " + str(label.get("x")) + " " + str(label.get("y")) + ")",
        )
        label.text = text
        return label

    # 値からy座標を求める
    def get_y(self, value: float):
        return self.get_position(value).y


class Data:
    def __init__(self, data_str: str, row_delim="\n", col_delim="\t"):
        self.data = []
        self.rows_unm = 0
        self.cols_num = 0
        rows = data_str.splitlines()
        rows = data_str.split(row_delim)
        for row_str in rows:
            if len(row_str) == 0 or row_str[0] == "#":
                # 空行、先頭が'#'の行は無視
                continue

            str_row = row_str.split(col_delim)
            row = []
            for str_el in str_row:
                num = None
                try:
                    num = float(str_el)
                except ValueError:
                    # 数値に変換できない場合はNone
                    num = None
                row.append(num)
            self.data.append(row)

            if self.cols_num < len(row):
                self.cols_num = len(row)

        self.rows_unm = len(self.data)

    def get(self):
        return self.data

    def get_size(self):
        return [self.rows_unm, self.cols_num]


class PlotData:
    def __init__(
        self, data: Data, x_column: int, y_column: int, x_axis: XAxis, y_axis: YAxis
    ):
        self.plot_data = []
        self.size = data.get_size()
        for i in range(len(data.get())):
            # None（数値に変換できなかったと思われる）がある行は無視
            if x_column >= len(data.get()[i]) or data.get()[i][x_column] == None:
                continue
            if y_column >= len(data.get()[i]) or data.get()[i][y_column] == None:
                continue
            x = data.get()[i][x_column]
            y = data.get()[i][y_column]
            self.plot_data.append([x, y])

        self.x_axis = x_axis
        self.y_axis = y_axis

    def get(self):
        return self.plot_data

    def get_points(self, shape_id=1):
        points = inkex.Group()
        for point_data in self.plot_data:
            x_value = point_data[0]
            y_value = point_data[1]

            # x, yが範囲外の場合は無視
            if self.x_axis.min > x_value or self.x_axis.max < x_value:
                continue
            if self.y_axis.min > y_value or self.y_axis.max < y_value:
                continue

            # x, yの値から描画する座標を求める
            pos_x = self.x_axis.get_x(x_value)
            pos_y = self.y_axis.get_y(y_value)

            point_element = GeneratePointEl.get(
                shape_id, inkex.Vector2d(pos_x, pos_y), 2
            )
            points.add(point_element)
        return points


class GeneratePointEl:
    shpes = [
        None,
        {"type": "circle", "args": {}, "style": "stroke"},
        {"type": "circle", "args": {}, "style": "fill"},
        {"type": "polygon", "args": {"sides": 4, "angle": 45.0}, "style": "stroke"},
        {"type": "polygon", "args": {"sides": 4, "angle": 45.0}, "style": "fill"},
        {"type": "polygon", "args": {"sides": 4, "angle": 0.0}, "style": "stroke"},
        {"type": "polygon", "args": {"sides": 4, "angle": 0.0}, "style": "fill"},
        {"type": "polygon", "args": {"sides": 3, "angle": 0.0}, "style": "stroke"},
        {"type": "polygon", "args": {"sides": 3, "angle": 0.0}, "style": "fill"},
        {"type": "polygon", "args": {"sides": 3, "angle": 180.0}, "style": "stroke"},
        {"type": "polygon", "args": {"sides": 3, "angle": 180.0}, "style": "fill"},
        {"type": "cross", "args": {"angle": 45.0}, "style": "stroke"},
        {"type": "cross", "args": {"angle": 0.0}, "style": "stroke"},
    ]

    @staticmethod
    def get(shape_id: int, position: inkex.Vector2d, size: float):
        if shape_id < 0 or shape_id >= len(GeneratePointEl.shpes):
            raise IndexError("指定されたshape_idのshapeは存在しません。")
        if shape_id == 0:
            return None

        radius = size / 2
        shape_data = GeneratePointEl.shpes[shape_id]
        type = shape_data["type"]
        if type == "circle":
            point = inkex.Circle(cx=str(position.x), cy=str(position.y), r=str(radius))
        elif type == "polygon":
            point = GeneratePointEl.polygon(position, radius, **shape_data["args"])
        elif type == "cross":
            point = GeneratePointEl.cross(position, radius, **shape_data["args"])

        style = shape_data["style"]
        if style == "stroke":
            point.style = GraphPlotter.base_stroke_style.copy()
        elif style == "fill":
            point.style = GraphPlotter.base_fill_style.copy()

        return point

    @staticmethod
    def polygon(position: inkex.Vector2d, radius: float, sides: int, angle: float):
        if sides < 3:
            raise ValueError("sidesは3以上にしてください。")

        # 正多角形の内角 = 180 * (n-2) / n
        # fix_angle: １つの頂点が中心の真上に来るようにするための角度
        fix_angle = 90 - 180 * (sides - 2) / sides
        polygon = inkex.PathElement.star(
            center=position,
            radii=(str(radius), "0"),
            sides=sides,
            rounded=0,
            args=(math.radians(fix_angle + angle), 0),
            flatsided=True,
            pathonly=False,
        )
        return polygon

    @staticmethod
    def cross(position: inkex.Vector2d, radius: float, angle: float):
        cross = inkex.Group()
        for i in range(2):
            line = inkex.Line(
                x1=str(position.x + radius * cos(np.radians(angle + 90 * i))),
                y1=str(position.y + radius * sin(np.radians(angle + 90 * i))),
                x2=str(position.x + radius * cos(np.radians(angle + 180 + 90 * i))),
                y2=str(position.y + radius * sin(np.radians(angle + 180 + 90 * i))),
            )
            cross.add(line)
        return cross


class GraphPlotter(inkex.Effect):
    # 基本スタイル
    """
    stroke-widthやfont-sizeは、
    GraphPlotterのeffect()内でドキュメントのユーザー単位に応じた大きさに変換する。
    self.svg.viewport_to_unit()を呼び出す。
    """
    base_text_style = inkex.Style(
        {
            "font-size": "1",
            "font-family": "Arial",
            "font-weight": "normal",
            "font-style": "normal",
            "text-align": "center",
            "text-anchor": "middle",
            "fill": "#000000",
            "stroke": "none",
            "stroke-width": "none",
        }
    )
    base_fill_style = inkex.Style(
        {
            "fill": "#000000",
            "stroke": "#000000",
            "stroke-width": "1",
            "stroke-linecap": "butt",
            "stroke-linejoin": "round",
        }
    )
    base_stroke_style = inkex.Style(
        {
            "fill": "none",
            "stroke": "#000000",
            "stroke-width": "1",
            "stroke-linecap": "butt",
            "stroke-linejoin": "round",
        }
    )

    def __init__(self):
        inkex.Effect.__init__(self)

        self.arg_parser.add_argument("--tab", type=str)

        # データ
        self.arg_parser.add_argument("--data_text", type=str, default="")
        self.arg_parser.add_argument("--row_delim", type=str, default="lf")
        self.arg_parser.add_argument("--col_delim", type=str, default="tab")

        # 描画設定
        self.arg_parser.add_argument("--setting_tab", type=str)
        # x軸
        self.arg_parser.add_argument("--x_axis_min", type=float, default="0")
        self.arg_parser.add_argument("--x_axis_max", type=float, default="100")
        self.arg_parser.add_argument(
            "--x_axis_reverse", type=inkex.Boolean, default="False"
        )
        self.arg_parser.add_argument("--x_axis_placement", type=str, default="bottom")
        self.arg_parser.add_argument("--x_axis_position", type=float, default="0")
        self.arg_parser.add_argument("--x_label_text", type=str, default="")
        self.arg_parser.add_argument("--x_label_position", type=float, default="0")
        self.arg_parser.add_argument("--x_maintick_step", type=float, default="10")
        self.arg_parser.add_argument("--x_maintick_offset", type=float, default="0")
        self.arg_parser.add_argument("--x_subtick_step", type=float, default="0")
        self.arg_parser.add_argument("--x_subtick_offset", type=float, default="0")
        self.arg_parser.add_argument("--x_number_step", type=float, default="50")
        self.arg_parser.add_argument("--x_number_offset", type=float, default="0")
        # y軸
        self.arg_parser.add_argument("--y_axis_min", type=float, default="0")
        self.arg_parser.add_argument("--y_axis_max", type=float, default="100")
        self.arg_parser.add_argument(
            "--y_axis_reverse", type=inkex.Boolean, default="False"
        )
        self.arg_parser.add_argument("--y_axis_placement", type=str, default="left")
        self.arg_parser.add_argument("--y_axis_position", type=float, default="0")
        self.arg_parser.add_argument("--y_label_text", type=str, default="")
        self.arg_parser.add_argument("--y_label_position", type=float, default="0")
        self.arg_parser.add_argument("--y_maintick_step", type=float, default="10")
        self.arg_parser.add_argument("--y_maintick_offset", type=float, default="0")
        self.arg_parser.add_argument("--y_subtick_step", type=float, default="0")
        self.arg_parser.add_argument("--y_subtick_offset", type=float, default="0")
        self.arg_parser.add_argument("--y_number_step", type=float, default="50")
        self.arg_parser.add_argument("--y_number_offset", type=float, default="0")
        # 点・線
        self.arg_parser.add_argument("--x_column", type=int, default="1")
        self.arg_parser.add_argument("--y_column", type=int, default="2")
        self.arg_parser.add_argument("--point_shape", type=int, default="1")
        self.arg_parser.add_argument("--line_type", type=int, default="0")
        self.arg_parser.add_argument("--line_width", type=float, default="0")
        # タイトル
        self.arg_parser.add_argument("--title_text", type=str, default="")
        self.arg_parser.add_argument("--title_placement", type=str, default="bottom")
        self.arg_parser.add_argument("--title_position", type=float, default="0")
        self.arg_parser.add_argument("--frame_top", type=inkex.Boolean, default="True")
        # フレーム
        self.arg_parser.add_argument(
            "--frame_bottom", type=inkex.Boolean, default="True"
        )
        self.arg_parser.add_argument("--frame_left", type=inkex.Boolean, default="True")
        self.arg_parser.add_argument(
            "--frame_right", type=inkex.Boolean, default="True"
        )
        # 描画項目
        self.arg_parser.add_argument(
            "--render_x_axis", type=inkex.Boolean, default="True"
        )
        self.arg_parser.add_argument(
            "--render_y_axis", type=inkex.Boolean, default="True"
        )
        self.arg_parser.add_argument(
            "--render_plot_data", type=inkex.Boolean, default="True"
        )
        self.arg_parser.add_argument(
            "--render_other", type=inkex.Boolean, default="True"
        )
        # 描画ページ
        self.arg_parser.add_argument("--page", type=int, default="1")

    def add_element(self, target_el, new_el):
        if isinstance(target_el, inkex.elements._svg.SvgDocumentElement):
            target_el.append(new_el)
        elif isinstance(
            target_el, (inkex.elements._groups.Group, inkex.elements._groups.Layer)
        ):
            target_el.add(new_el)
        else:
            target_el.addnext(new_el)

    def effect(self):
        # 0から始まるように調整
        self.options.x_column -= 1
        self.options.y_column -= 1
        self.options.page -= 1

        # 基本スタイル
        # 単位をドキュメントごとのユーザー単位に変換
        __class__.base_text_style["font-size"] = self.svg.viewport_to_unit("20pt")
        __class__.base_fill_style["stroke-width"] = self.svg.viewport_to_unit("2px")
        __class__.base_stroke_style["stroke-width"] = self.svg.viewport_to_unit("2px")

        # レンダーする場所の取得とか
        page_bbox = None
        try:
            page_bbox = self.svg.get_page_bbox(self.options.page)
        except IndexError:
            self.debug(
                "エラー: ページが見つかりません。ページの番号を確認してください。ページの番号は1から始まります。"
            )
            return
        center = inkex.Vector2d(page_bbox.center_x, page_bbox.center_y)
        size = self.svg.viewport_to_unit("400px")

        # グラフの描画領域にBoundingBoxを設定
        bbox = inkex.Rectangle(
            x=str(center.x - size / 2),
            y=str(center.y - size / 2),
            width=str(size),
            height=str(size),
        ).bounding_box()
        layer = self.svg.get_current_layer()
        parent_group = inkex.Group()
        if self.options.title_text:
            parent_group.set_id(self.options.title_text)
        else:
            parent_group.set_id(make_id("graph"))
        layer.add(parent_group)

        frame_style = __class__.base_stroke_style.copy()
        frame_style["stroke-linecap"] = "square"

        # render other
        if self.options.render_other:
            if self.options.title_text:
                title_position = self.svg.viewport_to_unit(
                    f"{self.options.title_position}px"
                )
                if self.options.title_placement == "bottom":
                    y = bbox.bottom + title_position
                else:
                    y = bbox.top - title_position
                title = TextElement()
                title.text = self.options.title_text
                title.style = __class__.base_text_style
                title.set_position(center.x, y)
                title.set_id(make_id("title"))
                parent_group.add(title)

            if (
                self.options.frame_top
                or self.options.frame_bottom
                or self.options.frame_left
                or self.options.frame_right
            ):
                frame_group = inkex.Group()
                frame_group.set_id(make_id("frames"))

                if self.options.frame_top:
                    line = inkex.Line(
                        x1=str(bbox.left),
                        y1=str(bbox.top),
                        x2=str(bbox.right),
                        y2=str(bbox.top),
                    )
                    line.style = frame_style
                    line.set_id(make_id("top"))
                    frame_group.add(line)
                if self.options.frame_bottom:
                    line = inkex.Line(
                        x1=str(bbox.left),
                        y1=str(bbox.bottom),
                        x2=str(bbox.right),
                        y2=str(bbox.bottom),
                    )
                    line.style = frame_style
                    line.set_id(make_id("bottom"))
                    frame_group.add(line)
                if self.options.frame_left:
                    line = inkex.Line(
                        x1=str(bbox.left),
                        y1=str(bbox.top),
                        x2=str(bbox.left),
                        y2=str(bbox.bottom),
                    )
                    line.style = frame_style
                    line.set_id(make_id("left"))
                    frame_group.add(line)
                if self.options.frame_right:
                    line = inkex.Line(
                        x1=str(bbox.right),
                        y1=str(bbox.top),
                        x2=str(bbox.right),
                        y2=str(bbox.bottom),
                    )
                    line.style = frame_style
                    line.set_id(make_id("right"))
                    frame_group.add(line)

                    parent_group.add(frame_group)

        maintick_size = self.svg.viewport_to_unit("16px")
        subtick_size = self.svg.viewport_to_unit("10px")

        # render x axis
        x_axis_position = self.svg.viewport_to_unit(f"{self.options.x_axis_position}px")
        if self.options.x_axis_placement == "bottom":
            y = bbox.bottom + x_axis_position
        else:
            y = bbox.top - x_axis_position

        x_axis = XAxis(
            min=self.options.x_axis_min,
            max=self.options.x_axis_max,
            start_x=bbox.right if self.options.x_axis_reverse else bbox.left,
            end_x=bbox.left if self.options.x_axis_reverse else bbox.right,
            y=y,
            plasement=self.options.x_axis_placement,
        )
        if self.options.render_x_axis:
            axis_group = inkex.Group()
            axis_group.set_id(make_id("x_axis"))
            if self.options.x_label_text:
                pos = self.svg.viewport_to_unit(f"{self.options.x_label_position}px")
                label = x_axis.get_label(self.options.x_label_text, pos)
                label.set_id(make_id("label"))
                axis_group.add(label)
            if self.options.x_maintick_step > 0:
                ticks = x_axis.get_ticks(
                    step=self.options.x_maintick_step,
                    offset=self.options.x_maintick_offset,
                    size=maintick_size,
                    stroke_width=self.svg.viewport_to_unit("2px"),
                )
                ticks.set_id(make_id("mainticks"))
                axis_group.add(ticks)
            if self.options.x_subtick_step > 0:
                ticks = x_axis.get_ticks(
                    self.options.x_subtick_step,
                    self.options.x_subtick_offset,
                    size=subtick_size,
                    stroke_width=self.svg.viewport_to_unit("2px"),
                )
                ticks.set_id(make_id("subticks"))
                axis_group.add(ticks)
            if self.options.x_subtick_step > 0 or self.options.x_maintick_step > 0:
                line = inkex.Line(
                    x1=str(x_axis.start.x),
                    y1=str(x_axis.start.y),
                    x2=str(x_axis.end.x),
                    y2=str(x_axis.end.y),
                )
                line.style = frame_style
                axis_group.add(line)
            if self.options.x_number_step > 0:
                numbers = x_axis.get_numbers(
                    self.options.x_number_step, self.options.x_number_offset
                )
                numbers.set_id(make_id("numbers"))
                axis_group.add(numbers)
            parent_group.add(axis_group)

        # render y axis
        y_axis_position = self.svg.viewport_to_unit(f"{self.options.y_axis_position}px")
        if self.options.y_axis_placement == "left":
            x = bbox.left - y_axis_position
        else:
            x = bbox.right + y_axis_position
        y_axis = YAxis(
            min=self.options.y_axis_min,
            max=self.options.y_axis_max,
            start_y=bbox.top if self.options.y_axis_reverse else bbox.bottom,
            end_y=bbox.bottom if self.options.y_axis_reverse else bbox.top,
            x=x,
            plasement=self.options.y_axis_placement,
        )
        if self.options.render_y_axis:
            axis_group = inkex.Group()
            if self.options.y_label_text:
                pos = self.svg.viewport_to_unit(f"{self.options.y_label_position}px")
                label = y_axis.get_label(self.options.y_label_text, pos)
                label.set_id(make_id("label"))
                axis_group.add(label)
            axis_group.set_id(make_id("y_axis"))
            if self.options.y_maintick_step > 0:
                ticks = y_axis.get_ticks(
                    step=self.options.y_maintick_step,
                    offset=self.options.y_maintick_offset,
                    size=maintick_size,
                    stroke_width=self.svg.viewport_to_unit("2px"),
                )
                axis_group.add(ticks)
                ticks.set_id(make_id("mainticks"))
            if self.options.y_subtick_step > 0:
                ticks = y_axis.get_ticks(
                    step=self.options.y_subtick_step,
                    offset=self.options.y_subtick_offset,
                    size=subtick_size,
                    stroke_width=self.svg.viewport_to_unit("2px"),
                )
                ticks.set_id(make_id("subticks"))
                axis_group.add(ticks)
            if self.options.y_subtick_step > 0 or self.options.y_maintick_step > 0:
                line = inkex.Line(
                    x1=str(y_axis.start.x),
                    y1=str(y_axis.start.y),
                    x2=str(y_axis.end.x),
                    y2=str(y_axis.end.y),
                )
                line.style = frame_style
                axis_group.add(line)
            if self.options.y_number_step > 0:
                numbers = y_axis.get_numbers(
                    self.options.y_number_step, self.options.y_number_offset
                )
                numbers.set_id(make_id("numbers"))
                axis_group.add(numbers)
            parent_group.add(axis_group)

        # render plot_data
        if self.options.render_plot_data:
            r_dlm = delim_list[self.options.row_delim]
            c_dlm = delim_list[self.options.col_delim]
            self.data = Data(self.options.data_text, r_dlm, c_dlm)
            plot_data = PlotData(
                self.data,
                self.options.x_column,
                self.options.y_column,
                x_axis,
                y_axis,
            )
            if self.options.point_shape > 0:
                points = plot_data.get_points(self.options.point_shape)
                points.set_id(make_id("points"))
                parent_group.add(points)


if __name__ == "__main__":
    e = GraphPlotter()
    e.run()
