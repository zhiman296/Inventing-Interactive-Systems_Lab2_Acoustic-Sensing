# BACK ROW NAP
#
# You fell asleep in the back row of the classroom.
# Ring the bell to wake you up so you don't get caught by the teacher!
# 
# The teacher gets closer in 3 danger levels:
#     !    he turns around      -> RING once
#     !!   he walks down aisle  -> RING twice
#     !!!  he stands over you   -> RING three times
#
# Right count = you fake studying, he goes back to the board.
# Wrong count or too slow = he gets closer (or catches you).
# 3 lives in total. 
# Ringing when it is safe makes him turn around!
# 
# 

import math
import queue
import random
import time

import numpy as np
import pygame
import sounddevice as sd

# Ring Detector Settings
SAMPLE_RATE = 44100
FFT_SIZE = 1024

DEBUG = False  # True = print detection numbers for tuning.

MIN_RMS = 0.02  # minimum loudness.
ONSET_RATIO = 4.0  # how suddenly the sound starts.
HIGH_RATIO = 50.0  # 5-16 kHz must exceed background by this much.
MID_LOW_RATIO = 2.0  # 1-4 kHz energy must be at least 2× the 100-800 Hz energy.
CONFIRM_FRAMES = 4  # wait 4 frames (~92 ms) before deciding.
DECAY_RATIO = 0.3  # the sound must have dropped below 30% of its peak.

COOLDOWN = 0.08  # ignore new rings for this long after one.
CALIBRATION_TIME = 1.0  #learn the background noise.

# Game Settings
GROUP_WINDOW = 0.55  # rings closer together than this count as one group.
LIVES = 3

STAGE_TIME = [3.4, 3.8, 4.2]
STAGE_MIN = [2.4, 2.8, 3.2]
SHRINK = 0.96  #each successful patrol multiplies the time by 0.96.

WALK_SPEED = 130  # teacher speed going toward you (pixels/s).
RETURN_SPEED = 260  # teacher speed going back to the board.

# Ring Detector
FREQS = np.fft.rfftfreq(FFT_SIZE, d=1 / SAMPLE_RATE)
WINDOW = np.hanning(FFT_SIZE)

# Select the bins in each band
LOW_MASK = (FREQS >= 100) & (FREQS < 800)   
MID_MASK = (FREQS >= 1000) & (FREQS < 4000)     
HIGH_MASK = (FREQS >= 5000) & (FREQS < 16000)   

baseline_high = None
baseline_rms = None
calibration_data = []  # Collects the startup samples. 
CALIBRATION_FRAMES = int(CALIBRATION_TIME * SAMPLE_RATE / FFT_SIZE)

recent_rms = [0.0] * 3
candidate = None
last_ring_time = 0.0
calibrated = False

ring_queue = queue.SimpleQueue()


def get_features(audio):
    rms = np.sqrt(np.mean(audio ** 2))  #overall loudness.

    audio = audio - np.mean(audio)
    magnitude = np.abs(np.fft.rfft(audio * WINDOW))
    power = magnitude ** 2

    low = np.mean(power[LOW_MASK])
    mid = np.mean(power[MID_MASK])
    high = np.mean(power[HIGH_MASK])

    return rms, low, mid, high


def process_frame(audio):
    global baseline_high, baseline_rms, candidate
    global last_ring_time, calibrated

    rms, low, mid, high = get_features(audio)
    now = time.time()

    onset = rms / (np.mean(recent_rms) + 1e-6)
    recent_rms.pop(0)
    recent_rms.append(rms)

    # Calibration
    if baseline_high is None:
        calibration_data.append((high, rms))
        if len(calibration_data) >= CALIBRATION_FRAMES:
            baseline_high = np.mean([d[0] for d in calibration_data])
            baseline_rms = np.mean([d[1] for d in calibration_data])
            calibrated = True
        return

    # A possible ring is waiting to be confirmed
    if candidate is not None:
        candidate["frames"] += 1

        if candidate["frames"] <= 2:
            candidate["peak"] = max(candidate["peak"], rms)

        if candidate["frames"] >= CONFIRM_FRAMES:
            died_away = rms < DECAY_RATIO * candidate["peak"]

            if DEBUG:
                print(f"  confirm: rms={rms:.3f} peak={candidate['peak']:.3f} "
                      f"-> {'RING' if died_away else 'rejected (lasted too long)'}")

            if died_away:
                last_ring_time = now
                ring_queue.put((now, candidate["peak"]))

            candidate = None
        return

    # Look for the start of a ring
    high_ratio = high / (baseline_high + 1e-12)
    mid_low_ratio = mid / (low + 1e-12)

    loud = rms > MIN_RMS
    sudden = onset > ONSET_RATIO
    bright = high_ratio > HIGH_RATIO
    not_low = mid_low_ratio > MID_LOW_RATIO
    ready = (now - last_ring_time) > COOLDOWN

    if DEBUG and loud and sudden:
        print(f"rms={rms:.3f} onset={onset:.1f} "
              f"high_ratio={high_ratio:.0f} mid/low={mid_low_ratio:.1f} | "
              f"bright={bright} not_low={not_low}")

    if loud and sudden and bright and not_low and ready:
        candidate = {"frames": 0, "peak": rms}
        return

    # Update background only when quiet
    if rms < 2 * baseline_rms:
        baseline_high = 0.98 * baseline_high + 0.02 * high
        baseline_rms = 0.98 * baseline_rms + 0.02 * rms


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    process_frame(indata[:, 0])


# Layout & Colors
W, H = 960, 540
FLOOR_Y = 430  # everyone stands on this line.

BOARD_X = 200  # where the teacher writes.
STAGE_X = {1: 360, 2: 520, 3: 680}

DESK_LEFT, DESK_RIGHT, DESK_Y = 730, 842, 352

# Background
WALL = (166, 166, 166)
FLOOR = (198, 198, 198)
PILLAR = (184, 184, 184)
LINE = (96, 96, 96)     
BOARD_C = (74, 74, 74)

# Characters
INK = (18, 18, 18)          
WHITE = (250, 250, 250)  # heads, hands, chalk
HAT = (78, 78, 78)
SHOE_GREY = (96, 96, 96)  # teacher shoes.
SHOE_BROWN = (140, 88, 44)  # student shoes.

SALMON = (240, 132, 124)
SALMON_D = (188, 88, 84)
TAN = (226, 190, 140)
BELL = (236, 168, 64)
POINTER = (200, 130, 50)
CHAIR = (110, 110, 110)
RED = (214, 40, 40)  # in danger.

HEAD_R = 26  # big round heads are the Stickmin look.
LINE_W = 4  # thickness of every stick limb.

def limb(surface, color, a, b, width):
    """A thick line with round ends (stickman arm, leg, body)."""
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    pygame.draw.line(surface, color, a, b, width)
    pygame.draw.circle(surface, color, a, width // 2)
    pygame.draw.circle(surface, color, b, width // 2)


# Game Logic
class Game:
    def __init__(self):
        self.t = 0.0
        self.state = "title" 
        self.over_t = 0.0
        self.mic_until = 0.0
        self.arcs = []
        self.popups = []
        self.flash = 0.0
        self.reset()

    # Setup
    def reset(self):
        self.lives = LIVES
        self.round = 0  # only used to speed the game up.

        self.phase = "teach"  # teach, alert, caught, return.
        self.stage = 0  
        self.tx = float(BOARD_X)
        self.target_x = float(BOARD_X)
        self.speed = WALK_SPEED
        self.facing = -1           
        self.moving = False

        self.teach_left = 4.5
        self.stage_left = 0.0
        self.stage_total = 1.0
        self.caught_left = 0.0
        self.fake_left = 0.0

        self.student = "asleep"   
        self.group_count = 0
        self.group_last = 0.0
        self.grace_until = 0.0

    def start_game(self):
        self.reset()
        self.state = "play"
        self.grace_until = self.t + 0.9
        self.arcs = []
        self.popups = []

    def popup(self, text):
        self.popups.append({"text": text, "born": self.t})

    # Rings
    def on_ring(self):
        self.arcs.append(self.t)
        self.mic_until = self.t + 0.7

        if self.state == "title":
            self.start_game()
            return
        if self.state == "over":
            if self.t - self.over_t > 1.2:
                self.start_game()
            return
        if self.t < self.grace_until:
            return

        if self.phase == "teach":
            if self.teach_left > 0.4:
                self.teach_left = 0.35
                self.popup("too loud!")
        elif self.phase == "alert":
            self.group_count += 1
            self.group_last = self.t

    def on_group_done(self, count):
        if count == self.stage:
            self.success()
        else:
            self.popup("wrong count!")
            self.mistake()

    # Teacher Moves
    def new_teach_time(self):
        return max(1.2, 4.2 - 0.22 * self.round) + random.uniform(0, 1.6)

    def start_patrol(self):
        stage = 1
        if self.round >= 6 and random.random() < 0.2:
            stage = 3
        elif self.round >= 3 and random.random() < 0.3:
            stage = 2
        self.enter_stage(stage)

    def enter_stage(self, k):
        self.phase = "alert"
        self.stage = k
        self.stage_total = max(STAGE_MIN[k - 1],
                               STAGE_TIME[k - 1] * (SHRINK ** self.round))
        self.stage_left = self.stage_total
        self.group_count = 0
        self.target_x = float(STAGE_X[k])
        self.facing = 1

        dist = abs(self.target_x - self.tx)
        if dist > 1:
            arrive = min(dist / WALK_SPEED, self.stage_total * 0.55)
            self.speed = dist / max(arrive, 0.05)
        else:
            self.speed = WALK_SPEED

    def success(self):
        self.round += 1
        self.go_back(faking=True)

    def mistake(self):
        if self.stage >= 3:
            self.caught()
        else:
            self.enter_stage(self.stage + 1)

    def timeout(self):
        self.popup("too slow!")
        self.mistake()

    def caught(self):
        self.phase = "caught"
        self.stage = 3
        self.caught_left = 2.0
        self.lives -= 1
        self.flash = 1.0
        self.student = "awake"
        self.group_count = 0
        self.target_x = float(STAGE_X[3])
        self.speed = 380
        self.facing = 1

    def go_back(self, faking):
        self.phase = "return"
        self.stage = 0
        self.group_count = 0
        self.target_x = float(BOARD_X)
        self.speed = RETURN_SPEED
        self.facing = -1
        self.student = "faking"
        self.fake_left = 1.6 if faking else 1.2

    # Per-frame Update
    def update(self, dt):
        self.t += dt
        self.flash = max(0.0, self.flash - dt * 1.4)
        self.arcs = [a for a in self.arcs if self.t - a < 0.8]
        self.popups = [p for p in self.popups if self.t - p["born"] < 1.2]

        if self.state == "play":
            self.update_play(dt)

    def update_play(self, dt):
        dx = self.target_x - self.tx
        if abs(dx) > 1.5:
            step = self.speed * dt
            if abs(dx) <= step:
                self.tx = self.target_x
                self.moving = False
            else:
                self.tx += math.copysign(step, dx)
                self.moving = True
        else:
            self.tx = self.target_x
            self.moving = False

        if self.phase == "teach":
            self.teach_left -= dt
            if self.teach_left <= 0:
                self.start_patrol()

        elif self.phase == "alert":
            self.stage_left -= dt

            if (self.group_count > 0
                    and self.t - self.group_last > GROUP_WINDOW):
                count = self.group_count
                self.group_count = 0
                self.on_group_done(count)
            elif self.stage_left <= 0:
                self.timeout()

        elif self.phase == "caught":
            self.caught_left -= dt
            if self.caught_left <= 0:
                if self.lives <= 0:
                    self.state = "over"
                    self.over_t = self.t
                else:
                    self.go_back(faking=False)

        elif self.phase == "return":
            self.fake_left = max(0.0, self.fake_left - dt)
            if abs(self.tx - BOARD_X) < 1.5 and self.fake_left <= 0:
                self.phase = "teach"
                self.student = "asleep"
                self.teach_left = self.new_teach_time()
                self.facing = -1


# Drawing
class Renderer:
    def __init__(self, screen):
        self.screen = screen

        # Fonts
        sans = "comic sans ms,chalkboard se,marker felt,comicneue,arial"
        self.f_title = pygame.font.SysFont(sans, 32, bold=True)
        self.f_body = pygame.font.SysFont(sans, 18)
        self.f_small = pygame.font.SysFont(sans, 15)
        self.f_bang = pygame.font.SysFont(sans, 44, bold=True)
        self.f_bang_small = pygame.font.SysFont(sans, 30, bold=True)

        self.veil = pygame.Surface((W, H), pygame.SRCALPHA)
        self.veil.fill((198, 198, 198, 232))
        self.flash_layer = pygame.Surface((W, H))
        self.flash_layer.fill(RED)

    def text(self, font, string, color, pos, anchor="topleft"):
        surf = font.render(string, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    # Small Reusable Pieces
    def mitten(self, pos):
        """Round white hand with a thin outline."""
        p = (int(pos[0]), int(pos[1]))
        pygame.draw.circle(self.screen, WHITE, p, 6)
        pygame.draw.circle(self.screen, INK, p, 6, 2)

    def shoe(self, pos, facing, color):
        """Small oval shoe, nudged forward in the facing direction."""
        rect = pygame.Rect(0, 0, 22, 10)
        rect.center = (int(pos[0] + facing * 5), int(pos[1] - 3))
        pygame.draw.ellipse(self.screen, color, rect)
        pygame.draw.ellipse(self.screen, INK, rect, 2)

    def head(self, x, y, facing, look):
        """White circle, thick outline, and eyes drawn as short strokes."""
        s = self.screen
        x, y = int(x), int(y)
        pygame.draw.circle(s, WHITE, (x, y), HEAD_R)
        pygame.draw.circle(s, INK, (x, y), HEAD_R, 4)

        e1 = x + facing * 11  # first eye.
        e2 = x + facing * 20  # second eye, closer to the edge.

        if look in ("normal", "angry"):
            for ex in (e1, e2):
                pygame.draw.line(s, INK, (ex, y - 12), (ex, y + 6), 3)
            if look == "angry":
                for ex in (e1, e2):     # brows slope down toward the nose
                    pygame.draw.line(s, INK,
                                     (ex + facing * 5, y - 19),
                                     (ex - facing * 5, y - 13), 3)
        elif look == "sleep":
            for ex in (e1, e2):
                pygame.draw.line(s, INK, (ex - 4, y + 2), (ex + 4, y + 2), 3)
        elif look == "wide":
            for ex in (e1, e2):
                pygame.draw.circle(s, WHITE, (ex, y - 2), 6)
                pygame.draw.circle(s, INK, (ex, y - 2), 6, 2)
                pygame.draw.circle(s, INK, (ex, y - 2), 2)
        elif look == "read":
            pygame.draw.circle(s, INK, (e1, y - 3), 3)
            pygame.draw.circle(s, INK, (e2, y - 3), 3)
            pygame.draw.line(s, INK, (e1 - 3, y + 11), (e1 + 5, y + 11), 2)

    def hat(self, x, y, facing):
        """Dark hat with a little red badge, sitting on the teacher's head."""
        s = self.screen
        x, y = int(x), int(y)
        rect = pygame.Rect(x - HEAD_R - 2, y - HEAD_R - 12,
                           2 * HEAD_R + 4, 22)
        pygame.draw.rect(s, HAT, rect, border_radius=9)
        pygame.draw.rect(s, INK, rect, 3, border_radius=9)
        pygame.draw.circle(s, RED, (x + facing * 14, y - HEAD_R - 1), 4)

    # The Scene

    def draw(self, g):
        s = self.screen
        self.draw_scene()
        self.draw_board()
        self.draw_teacher(g)
        self.draw_desk_and_student(g)
        self.draw_arcs(g)
        self.draw_zzz(g)
        self.draw_bubble(g)
        self.draw_popups(g)

        if g.flash > 0:
            self.flash_layer.set_alpha(int(110 * g.flash))
            s.blit(self.flash_layer, (0, 0))

        self.draw_hud(g)

        if g.state == "title":
            self.draw_title(g)
        elif g.state == "over":
            self.draw_over(g)

    def draw_crate(self, rect):
        pygame.draw.rect(self.screen, SALMON, rect)
        pygame.draw.rect(self.screen, SALMON_D, rect, 3)
        mid = rect.centerx
        pygame.draw.line(self.screen, SALMON_D,
                         (mid, rect.top), (mid, rect.bottom), 3)

    def draw_scene(self):
        s = self.screen
        s.fill(WALL)
        pygame.draw.rect(s, FLOOR, (0, FLOOR_Y, W, H - FLOOR_Y))
        pygame.draw.line(s, LINE, (0, FLOOR_Y), (W, FLOOR_Y), 3)

        # A stack of muted salmon crates in the corner.
        self.draw_crate(pygame.Rect(-10, 372, 96, 58))
        self.draw_crate(pygame.Rect(6, 318, 70, 54))

        # Dust dots on the floor
        for p in [(130, 468), (146, 474), (520, 492), (610, 470)]:
            pygame.draw.circle(s, LINE, p, 2)

    def draw_board(self):
        s = self.screen
        board = pygame.Rect(60, 60, 500, 240)
        pygame.draw.rect(s, BOARD_C, board)
        pygame.draw.rect(s, INK, board, 4)
        pygame.draw.line(s, INK, (60, 306), (560, 306), 5) 

        # chalk lines with a slight hand-drawn wobble
        for i, length in enumerate([260, 200, 300, 150]):
            y = 110 + i * 40
            pts = [(100 + k, y + math.sin(k * 0.35 + i) * 2)
                   for k in range(0, length + 1, 8)]
            pygame.draw.lines(s, WHITE, False, pts, 3)

    def draw_teacher(self, g):
        s = self.screen
        f = g.facing
        x = g.tx
        hip = (x, FLOOR_Y - 74)
        sh_y = FLOOR_Y - 140
        shoulder = (x, sh_y)
        head = (x + f * 3, sh_y - 30)

        fast = g.speed > 200
        swing = math.sin(g.t * (15 if fast else 10))

        if g.moving:
            pose = "walk"
        elif g.phase == "teach" or g.state != "play":
            pose = "write"
        elif g.phase == "caught":
            pose = "point"
        else:
            pose = "stand"

        # Legs and shoes
        if pose == "walk":
            foot_a = (x + swing * 30, FLOOR_Y - max(0.0, swing) * 10)
            foot_b = (x - swing * 30, FLOOR_Y - max(0.0, -swing) * 10)
        else:
            foot_a = (x - 14, FLOOR_Y)
            foot_b = (x + 14, FLOOR_Y)
        limb(s, INK, hip, foot_a, LINE_W)
        limb(s, INK, hip, foot_b, LINE_W)
        self.shoe(foot_a, f, SHOE_GREY)
        self.shoe(foot_b, f, SHOE_GREY)

        # Body
        limb(s, INK, hip, shoulder, LINE_W)

        # Arms, pointer and hands
        arm_top = (x, sh_y + 8)
        if pose == "walk":
            free_hand = (x - swing * 24, sh_y + 58)
            hand = (x + swing * 24, sh_y + 58)
            tip = (hand[0] + f * 46, hand[1] + 30)
        elif pose == "write":
            free_hand = (x - f * 6, sh_y + 60)
            wiggle = math.sin(g.t * 6) * 8
            hand = (x + f * 40 + wiggle * f, sh_y - 26 + wiggle * 0.5)
            tip = (hand[0] + f * 46, hand[1] - 46)
        elif pose == "point":
            free_hand = (x - f * 6, sh_y + 60)
            hand = (x + f * 62, sh_y + 6)
            tip = (hand[0] + f * 86, hand[1] + 12)
        else:
            free_hand = (x + f * 10, sh_y + 62)
            hand = (x + f * 34, sh_y + 52)
            tip = (hand[0] + f * 46, hand[1] + 30)

        limb(s, INK, arm_top, free_hand, LINE_W)
        limb(s, INK, arm_top, hand, LINE_W)
        limb(s, POINTER, hand, tip, 3)
        self.mitten(free_hand)
        self.mitten(hand)

        # Head
        angry = (g.phase == "alert" and g.stage >= 2) or g.phase == "caught"
        self.head(head[0], head[1], f, "angry" if angry else "normal")

    def draw_desk_and_student(self, g):
        s = self.screen

        # chair (behind the student)
        limb(s, CHAIR, (846, 388), (902, 388), 5)
        limb(s, CHAIR, (900, 388), (900, 318), 5)
        limb(s, CHAIR, (854, 388), (854, FLOOR_Y), 5)
        limb(s, CHAIR, (896, 388), (896, FLOOR_Y), 5)

        # The student's legs
        hip = (876, 384)
        knee = (820, 384)
        limb(s, INK, hip, knee, LINE_W)
        limb(s, INK, knee, (820, FLOOR_Y - 4), LINE_W)
        self.shoe((820, FLOOR_Y), -1, SHOE_BROWN)

        if g.student == "asleep":
            bob = math.sin(g.t * 2.2) * 2
            shoulder = (850, 338 + bob)
            head = (818, 322 + bob)
            limb(s, INK, hip, shoulder, LINE_W)
            limb(s, INK, shoulder, (796, 347), LINE_W)  # folded arms.
            self.mitten((796, 347))
            self.head(head[0], head[1], -1, "sleep")
        else:
            shoulder = (876, 320)
            head = (872, 294)
            limb(s, INK, hip, shoulder, LINE_W)
            limb(s, INK, shoulder, (824, 346), LINE_W)  # arm on desk.
            self.mitten((824, 346))
            if g.student == "awake":
                self.head(head[0], head[1], -1, "wide")
                drop = int(g.t * 6) % 3 * 3
                d = (int(head[0] + 34), int(head[1] - 20 + drop))
                pygame.draw.circle(s, (170, 205, 235), d, 4)
                pygame.draw.circle(s, INK, d, 4, 1)
            else:
                # An open book on the desk.
                book = pygame.Rect(782, 340, 40, 12)
                pygame.draw.rect(s, WHITE, book)
                pygame.draw.rect(s, INK, book, 2)
                pygame.draw.line(s, INK, (802, 340), (802, 352), 2)
                self.head(head[0], head[1], -1, "read")

        # Desk
        desk = pygame.Rect(DESK_LEFT, DESK_Y, DESK_RIGHT - DESK_LEFT, 12)
        pygame.draw.rect(s, TAN, desk)
        pygame.draw.rect(s, INK, desk, 3)
        limb(s, INK, (742, DESK_Y + 12), (742, FLOOR_Y), 5)
        limb(s, INK, (830, DESK_Y + 12), (830, FLOOR_Y), 5)

    def draw_arcs(self, g):
        s = self.screen
        for ta in g.arcs:
            age = g.t - ta
            fade = max(0.0, 1.0 - age / 0.8)
            color = tuple(int(WALL[i] * (1 - fade) + INK[i] * fade)
                          for i in range(3))
            for k in range(2):
                r = 20 + (age - k * 0.12) * 110
                if r > 20:
                    rect = (758 - r, DESK_Y - 8 - r, 2 * r, 2 * r)
                    pygame.draw.arc(s, color, rect, 0.6, math.pi - 0.6, 3)

    def draw_zzz(self, g):
        if g.student != "asleep" or g.state == "over":
            return
        for i in range(3):
            p = (g.t * 0.45 + i / 3) % 1.0
            x = 830 + p * 40
            y = 285 - p * 90
            surf = self.f_bang_small.render("z", True, (60, 60, 60))
            surf = pygame.transform.rotozoom(surf, 0, 0.5 + 0.7 * p)
            surf.set_alpha(int(255 * (1 - p)))
            self.screen.blit(surf, surf.get_rect(center=(x, y)))

    def draw_bubble(self, g):
        if g.state != "play":
            return
        if g.phase == "alert":
            label = "!" * g.stage
        elif g.phase == "caught":
            label = "!!!"
        else:
            return

        bounce = math.sin(g.t * (8 + 3 * max(g.stage, 1))) * 3
        surf = self.f_bang.render(label, True, RED)
        bw, bh = surf.get_width() + 26, surf.get_height() + 6
        cx = int(g.tx)
        # Sits above the hat: shoulder line, head offset, hat height,gap.
        cy = FLOOR_Y - 140 - 30 - 38 - 20 - bh // 2 + int(bounce)

        rect = pygame.Rect(0, 0, bw, bh)
        rect.center = (cx, cy)
        rect.clamp_ip(self.screen.get_rect())

        tail_x = max(rect.left + 14, min(rect.right - 14, cx))
        tail = [(tail_x - 9, rect.bottom - 2), (tail_x + 9, rect.bottom - 2),
                (tail_x, rect.bottom + 13)]
        pygame.draw.rect(self.screen, WHITE, rect, border_radius=10)
        pygame.draw.polygon(self.screen, WHITE, tail)
        pygame.draw.rect(self.screen, INK, rect, 3, border_radius=10)
        pygame.draw.lines(self.screen, INK, False,
                          [tail[0], tail[2], tail[1]], 3)
        pygame.draw.line(self.screen, WHITE, (tail[0][0] + 2, rect.bottom - 1),
                         (tail[1][0] - 2, rect.bottom - 1), 4)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def draw_popups(self, g):
        for p in g.popups:
            age = g.t - p["born"]
            surf = self.f_body.render(p["text"], True, RED)
            surf.set_alpha(int(255 * max(0.0, 1 - age / 1.2)))
            self.screen.blit(surf, surf.get_rect(center=(800, 250 - age * 30)))

    # HUD
    def draw_heart(self, x, y, filled):
        pattern = [".XX.XX.",
                   "XXXXXXX",
                   "XXXXXXX",
                   ".XXXXX.",
                   "..XXX..",
                   "...X..."]
        color = RED if filled else (130, 130, 130)
        for row, line in enumerate(pattern):
            for col, ch in enumerate(line):
                if ch == "X":
                    pygame.draw.rect(self.screen, color,
                                     (x + col * 4, y + row * 4, 4, 4))

    def draw_hud(self, g):
        s = self.screen

        HEART_W = 28       
        GAP = 10             
        MARGIN = 24           
        for i in range(LIVES):
            x = W - MARGIN - HEART_W - i * (HEART_W + GAP)
            self.draw_heart(x, 22, i < g.lives)

        if not calibrated:
            mic = "mic: calibrating..."
        elif g.t < g.mic_until:
            mic = "mic: Ring"
        else:
            mic = "mic: No Interaction"
        self.text(self.f_small, mic, (60, 60, 60), (24, H - 30))

        if g.state == "play" and g.phase == "alert":
            ratio = max(0.0, min(1.0, g.stage_left / g.stage_total))
            bar = pygame.Rect(W // 2 - 150, H - 64, 300, 12)
            pygame.draw.rect(s, (150, 150, 150), bar, border_radius=6)
            inner = bar.copy()
            inner.width = int(bar.width * ratio)
            pygame.draw.rect(s, INK, inner, border_radius=6)
            pygame.draw.rect(s, INK, bar, 2, border_radius=6)

            shown = max(g.stage, g.group_count)
            x0 = W // 2 - (shown - 1) * 18
            for i in range(shown):
                center = (x0 + i * 36, H - 32)
                if i >= g.stage:
                    pygame.draw.circle(s, RED, center, 9)
                elif i < g.group_count:
                    pygame.draw.circle(s, INK, center, 9)
                else:
                    pygame.draw.circle(s, WHITE, center, 9)
                    pygame.draw.circle(s, INK, center, 9, 2)

    # Title and game over
    def draw_title(self, g):
        self.screen.blit(self.veil, (0, 0))
        cx = W // 2
        self.text(self.f_title, "BACK ROW NAP", INK, (cx, 110), "midtop")
        self.text(self.f_body,
                  "You fell asleep in class. Ring to wake up before the "
                  "teacher catches you.", (60, 60, 60), (cx, 166), "midtop")

        rows = [("!", "ring once"), ("!!", "ring twice"),
                ("!!!", "ring three times")]
        for i, (bang, label) in enumerate(rows):
            y = 226 + i * 40
            self.text(self.f_bang_small, bang, RED, (cx - 24, y), "topright")
            self.text(self.f_body, label, INK, (cx, y + 8))

        self.text(self.f_small,
                  f"{LIVES} lives  -  don't ring when it's safe",
                  (70, 70, 70), (cx, 356), "midtop")

        if not calibrated:
            self.text(self.f_body, "calibrating... stay quiet", (70, 70, 70),
                      (cx, 408), "midtop")
        else:
            self.text(self.f_body, "ring to start", INK,
                      (cx, 408), "midtop")

    def draw_over(self, g):
        self.screen.blit(self.veil, (0, 0))
        cx = W // 2
        self.text(self.f_title, "BUSTED", RED, (cx, 200), "midtop")
        if g.t - g.over_t > 1.2:
            self.text(self.f_body, "ring to try again", INK,
                      (cx, 256), "midtop")


# Main Loop
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Back Row Nap")
    clock = pygame.time.Clock()

    renderer = Renderer(screen)
    game = Game()

    print("Starting microphone...")
    print("Stay quiet for 1 second (calibrating)...")
    print("Ring to play. SPACE = test ring. ESC = quit.")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=FFT_SIZE, callback=audio_callback):
        running = True
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # elif event.key == pygame.K_SPACE:
                    #     ring_queue.put((time.time(), 0.6))   # test ring

            while True:
                try:
                    ring_queue.get_nowait()
                except queue.Empty:
                    break
                print("Ring")
                game.on_ring()

            game.update(dt)
            renderer.draw(game)
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()