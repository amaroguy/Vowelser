import pygame
import numpy as np

class FormantTracker:
    def __init__(self, WIDTH, HEIGHT, F2_RANGE, F1_RANGE):
        
        pygame.init()

        self.WIDTH, self.HEIGHT = WIDTH, HEIGHT
        self.F1_RANGE, self.F2_RANGE = F1_RANGE, F2_RANGE
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        pygame.display.set_caption("Formants!")

    def draw_formant(self, f2, f1):

        if f1 < self.F1_RANGE[0] or f1 > self.F1_RANGE[1]:
            # print(f"f1: {f1} not in range ${self.F1_RANGE}")
            return
        if f2 < self.F2_RANGE[0] or f2 > self.F2_RANGE[1]:
            # print(f"f2: {f1} not in range ${self.F2_RANGE}")
            return

        x = np.interp(f1, [self.F1_RANGE[0], self.F1_RANGE[1]], [0, self.WIDTH])  
        y = np.interp(f2, [self.F2_RANGE[0], self.F2_RANGE[1]], [self.HEIGHT, 0])
        y, x = int(x), int(y)

        self.screen.fill((0,0,0))
        pygame.draw.circle(self.screen, (255,0,0), (x,y), 10)
        pygame.display.flip()
    
    def destroy(self):
        pygame.quit()