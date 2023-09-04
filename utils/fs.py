import cv2
import insightface
import datetime
from insightface.app import FaceAnalysis
from file_server import start_server

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(
    r'D:\newpose\utils\inswapper_128.onnx', download=False, download_zip=False)


def faceswap(img2_fn):
    '''
        faceswap(배경이미지주소, 얼굴이미지주소, 얼굴분석모델(app), 스왑모델(swapper))

        pip install insightface, images라는 이미지 담는 디렉토리 만들어주기
    '''

    img_arr = [
        'images/1.png', 'images/am.png', 'images/front1.jpg'
    ]

    new_img_arr = []

    for img in img_arr:
        img = cv2.imread(img)
        faces = app.get(img)
        print("detected number of faces: ", len(faces))
        dn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fn = f'fs_{dn}.jpg'
        filepathname = f'images/{fn}'
        img2 = cv2.imread(img2_fn)
        face2 = app.get(img2)[0]

        for face in faces:
            img = swapper.get(img, face, face2, paste_back=True)

        new_img_arr.append(filepathname)
        cv2.imwrite(f'images/{fn}', img)
        print(f"saved a file successfully. {fn}")

    # with open(f'images/{fn}', 'rb') as openfile:
        # 유니티 포톤의 url
        # requests.post('http://localhost:5000/files/upload',
        #               files={"newfile": openfile})

    # tcp_send_thread = threading.Thread(
    #     target=start_server, args=('192.168.1.24', 5656, new_img_arr))
    # tcp_send_thread.daemon = True
    # tcp_send_thread

    # treat_img_arr(new_img_arr)
    start_server('192.168.1.63', 5656, new_img_arr[0])
    print("send file to unity successfully")
    return None


if __name__ == '__main__':
    faceswap('images/ic1.jpg')
