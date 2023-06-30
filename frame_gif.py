import os

import cv2
import imageio


def main():
    img_file = os.listdir('./dataset/')
    img_file.sort(key=lambda x: int(x[7:-4]))
    img_group = []
    for i, file in enumerate(img_file):
        if i % 1 == 0:
            img = cv2.imread('./dataset/' + file, 3)
            # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            img_group.append(img)
    imageio.mimsave('./result.gif', img_group, 'GIF', duration=1e-3)


if __name__ == '__main__':
    main()
