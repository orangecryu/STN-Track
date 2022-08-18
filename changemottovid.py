import cv2
import glob


def resize(img_array, align_mode):
    _height = len(img_array[0])
    _width = len(img_array[0][0])
    for i in range(1, len(img_array)):
        img = img_array[i]
        height = len(img)
        width = len(img[0])
        if align_mode == 'smallest':
            if height < _height:
                _height = height
            if width < _width:
                _width = width
        else:
            if height > _height:
                _height = height
            if width > _width:
                _width = width

    for i in range(0, len(img_array)):
        img1 = cv2.resize(img_array[i], (_width, _height), interpolation=cv2.INTER_CUBIC)
        img_array[i] = img1

    return img_array, (_width, _height)


def images_to_video(path):
    img_array = []

    for filename in glob.glob(path + '/*.jpg'):
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    # 图片的大小需要一致
    img_array, size = resize(img_array, 'largest')
    fps = 15
    out = cv2.VideoWriter('uav0000120_04775_v.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def main():
    path = r"D:\xxk\ByteTrack-main\datasets\visdrone\train\uav0000120_04775_v\img1"
    images_to_video(path)
# r"D:\xxk\UAV-benchmark-M\M0205\img1"

if __name__ == "__main__":
    main()

