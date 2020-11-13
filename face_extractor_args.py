import video_face_extractor as fe
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse


def show_image(file):

    img = cv2.imread(file)

    all_faces, full_bgr_img = fe.detect_faces(img)

    if len(all_faces) > 0:
        faces = []
        for face in all_faces:
            faces.append(cv2.resize(face, (200, 280),
                                    interpolation=cv2.INTER_AREA))
        e_faces = np.hstack(faces)
        cv2.imshow("Faces", e_faces)

    # cv2.imshow("My img", full_bgr_img)
    plt.imshow(cv2.cvtColor(full_bgr_img, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_video(video):
    cap = cv2.VideoCapture(video)

    while True:
        rect, frame = cap.read(0)
        # my_frame = cv2.resize(frame,None, fx=0.50, fy=0.5)
        my_frame = cv2.flip(frame, 1)
        f = fe.detect_faces(my_frame)

        cv2.imshow("Full video", f[1])

        # append faces
        if len(f[0]) > 0:
            faces = []
            for face in f[0]:
                faces.append(cv2.resize(face, (200, 280),
                                        interpolation=cv2.INTER_AREA))
            all_faces = np.hstack(faces)
            cv2.imshow("Faces", all_faces)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Face extractor")
    parser.add_argument("path", type=str,
                        help="path of image you want to extract face from")
    args = parser.parse_args()

    if args.path.endswith(".jpg") or args.path.endswith(".png") or args.path.endswith(".jpeg"):
        show_image(args.path)
    elif args.path.endswith(".mp4"):
        show_video(args.path)
    else:
        print("File format not supported")
