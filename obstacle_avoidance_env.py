import numpy as np
import pyglet

pyglet.clock.set_fps_limit(10000)

def three_point_distance(l_point1, l_point2, target):
    l_point1, l_point2, target = np.append(l_point1, 0), np.append(l_point2, 0),\
                                 np.append(target, 0)
    v1 = target - l_point1
    v2 = target - l_point2
    line = l_point1 - l_point2

    area = np.linalg.norm(np.cross(v1, v2))
    if area >= 1:
        return area / (np.linalg.norm(line) + 1e-10)
    else:
        p1_dis = np.linalg.norm(v1)
        p2_dis = np.linalg.norm(v2)
        if max(p1_dis,p2_dis) > 100:
            return min(p1_dis, p2_dis)
        else:
            return 0.0

class ArmEnv(object):
    action_bound = [-1, 1]
    action_dim = 3
    state_dim = 15
    dt = .1  # refresh rate
    arm1l = 100
    arm2l = 100
    arm3l = 100
    obstacle_ratio = 15
    viewer = None
    viewer_xy = (600, 600)
    get_point = False
    mouse_in = np.array([False])
    point_l = 15
    grab_counter = 0

    def __init__(self, mode='easy'):
        self.mode = mode
        self.arm_info = np.zeros((3,4))
        self.arm_info[0, 0] = self.arm1l
        self.arm_info[1, 0] = self.arm2l
        self.arm_info[2, 0] = self.arm3l
        self.point_info = np.array([500, 500])
        self.obstacle_info = np.array([250, 303])
        self.obstacle_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_xy) / 2
        self.dis_old = np.array([0, 0])

    def step(self, action):
        _, self.dis_old, _ = self._get_state()

        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, *self.action_bound)
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2

        arm1rad = self.arm_info[0, 1]
        arm2rad = self.arm_info[1, 1]
        arm3rad = self.arm_info[2, 1]
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
        arm3dx_dy = np.array([self.arm_info[2, 0] * np.cos(arm3rad), self.arm_info[2, 0] * np.sin(arm3rad)])
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)
        self.arm_info[2, 2:4] = self.arm_info[1, 2:4] + arm3dx_dy

        s, arm2_distance, arm_distance = self._get_state()
        r = self._r_func(self.dis_old, arm2_distance, arm_distance)

        return s, r, self.get_point

    def _get_state(self):
        arm_end = self.arm_info[:, 2:4]
        # 3x4 matrix
        t_arms = np.ravel(arm_end - self.point_info)
        # 3D vector
        o_arms = np.ravel(arm_end - self.obstacle_info)
        o_dis = self.obstacle_to_arms(arm_end, self.obstacle_info)
        center_dis = (self.center_coord - self.obstacle_info)/200
        in_point = 1 if self.grab_counter > 0 else 0
        return np.hstack([in_point, t_arms/200, o_arms/200, center_dis]), t_arms[-2:], o_dis

    def _r_func(self, dis_old, distance, distance2):
        t = 20
        abs_dis_old = np.sqrt(np.sum(np.square(dis_old)))
        # arms end to the target
        abs_distance1 = np.sqrt(np.sum(np.square(distance)))
        # r = -abs_distance1 / 600
        abs_distance2 = np.sqrt(np.min(np.square(distance2)))
        r = 0
        if 15 < abs_distance2 < 30:
            punishiment = (3 * (30 - abs_distance2) / 15) + 2
            r -= punishiment
        if abs_distance2 < 15:
            punishiment = (40 * (15 - abs_distance2) / 15) + 10
            r -= punishiment

        if abs_dis_old > abs_distance1:
            r += 1
        else:
            r -= 1

        if abs_distance1 < self.point_l and (not self.get_point):
            r += 10
            self.grab_counter += 1
            if self.grab_counter > t:
                r += 20
                self.get_point = True
        elif abs_distance1 > self.point_l:
            self.grab_counter = 0
            self.get_point = False
        return r

    def obstacle_to_arms(self, arm, obstacle):
        dis = []
        arms = np.insert(arm, 0, [0, 0], axis=0)
        for i in range(arm.shape[0]):
            dis.append(three_point_distance(arms[i], arms[i+1], obstacle))
        return np.asarray(dis)

    def reset(self):
        self.get_point = False
        self.grab_counter = 0

        if self.mode == 'hard':
            pxy = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 500)
            self.obstacle_info[:] = pxy
        else:
            arm1rad, arm2rad, arm3rad = np.random.rand(3) * np.pi * 2
            self.arm_info[0, 1] = arm1rad
            self.arm_info[1, 1] = arm2rad
            self.arm_info[2, 1] = arm3rad
            arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
            arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
            arm3dx_dy = np.array([self.arm_info[2, 0] * np.cos(arm3rad), self.arm_info[2, 0] * np.sin(arm3rad)])
            self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
            self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)
            self.arm_info[2, 2:4] = self.arm_info[1, 2:4] + arm3dx_dy

            self.obstacle_info[:] = np.random.uniform([-200., -200.], [200., 200.]) + self.center_coord
            # print(self.point_info)
        return self._get_state()[0]

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.obstacle_info, self.mouse_in)
        self.viewer.render()

    def set_fps(self, fps):
        pyglet.clock.set_fps_limit(fps)

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, arm_info, point_info, point_l, obstacle_info, mouse_in):
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.point_info = point_info
        self.mouse_in = mouse_in
        self.point_l = point_l
        self.obstacle_info = obstacle_info

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()

        arm1_box, arm2_box, arm3_box, point_box, obstacle_box = [0]*8, [0]*8, [0]*8, [0]*8, [0]*8
        c1, c2, c3 = (249, 86, 86)*4, (86, 109, 249)*4, (249, 39, 65)*4
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm1_box), ('c3B', c1))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm2_box), ('c3B', c1))
        self.arm3 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm3_box), ('c3B', c1))
        self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm3_box), ('c3B', c1))

    def render(self):
        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update_arm(self):
        point_l = self.point_l
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box

        obstacle_box = (self.obstacle_info[0] - point_l, self.obstacle_info[1] - point_l,
                     self.obstacle_info[0] + point_l, self.obstacle_info[1] - point_l,
                     self.obstacle_info[0] + point_l, self.obstacle_info[1] + point_l,
                     self.obstacle_info[0] - point_l, self.obstacle_info[1] + point_l)
        self.obstacle.vertices = obstacle_box

        arm1_coord = (*self.center_coord, *(self.arm_info[0, 2:4]))  # (x0, y0, x1, y1)
        arm2_coord = (*(self.arm_info[0, 2:4]), *(self.arm_info[1, 2:4]))  # (x1, y1, x2, y2)
        arm3_coord = (*(self.arm_info[1, 2:4]), *(self.arm_info[2, 2:4]))
        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 1]
        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(
            arm1_thick_rad) * self.bar_thc
        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(
            arm1_thick_rad) * self.bar_thc
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)
        arm2_thick_rad = np.pi / 2 - self.arm_info[1, 1]
        x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
            arm2_thick_rad) * self.bar_thc
        x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
            arm2_thick_rad) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)
        arm3_thick_rad = np.pi / 2 - self.arm_info[2, 1]
        x11__, y11__ = arm3_coord[0] + np.cos(arm3_thick_rad) * self.bar_thc, arm3_coord[1] - np.sin(
            arm3_thick_rad) * self.bar_thc
        x12__, y12__ = arm3_coord[0] - np.cos(arm3_thick_rad) * self.bar_thc, arm3_coord[1] + np.sin(
            arm3_thick_rad) * self.bar_thc
        x21__, y21__ = arm3_coord[2] - np.cos(arm3_thick_rad) * self.bar_thc, arm3_coord[3] + np.sin(
            arm3_thick_rad) * self.bar_thc
        x22__, y22__ = arm3_coord[2] + np.cos(arm3_thick_rad) * self.bar_thc, arm3_coord[3] - np.sin(
            arm3_thick_rad) * self.bar_thc
        arm3_box = (x11__, y11__, x12__, y12__, x21__, y21__, x22__, y22__)
        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box
        self.arm3.vertices = arm3_box

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm_info[0, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.W:
            self.arm_info[2, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.E:
            self.arm_info[2, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.obstacle_info[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False

if __name__ == '__main__':
    env = ArmEnv()
    s = env.reset()
    print(s.shape)
