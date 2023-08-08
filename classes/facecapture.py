class facecapture:

    img_rgb=None
    results=None

    def __init__(self,face_mesh):
        self.face_mesh=face_mesh

    def faces_process_get(self):
        while True:
            if (self.img_rgb is not None):
                self.get_crop_from_faces()


    def get_crop_from_faces(self):
        self.results = self.face_mesh.process(self.img_rgb)
       
    def update_img(self,img_rgb):
        self.img_rgb=img_rgb