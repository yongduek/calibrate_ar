from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from PIL import Image
import cv2
import numpy as np
import imutils
import sys

def extrinsic2ModelView(RVEC, TVEC, R_vector = True):
    """[Get modelview matrix from RVEC and TVEC]

    Arguments:
        RVEC {[vector]} -- [Rotation vector]
        TVEC {[vector]} -- [Translation vector]
    """ 

    R, _ = cv2.Rodrigues(RVEC)
    
    ## OpenCV to OpenGL
    Rx = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    TVEC = TVEC.flatten().reshape((3, 1))

    cvPose = np.hstack((R, TVEC))
    transform_matrix = Rx @ cvPose
    M = np.eye(4)
    M[:3, :] = transform_matrix

    return M.T.flatten()


def intrinsic2Project(MTX, width, height, near_plane=0.01, far_plane=100.0):
    P = np.zeros(shape=(4, 4), dtype=np.float32)
    
    fx, fy = MTX[0, 0], MTX[1, 1]
    cx, cy = MTX[0, 2], MTX[1, 2]
    
    
    P[0, 0] = 2 * fx / width
    P[1, 1] = 2 * fy / (height)
    P[2, 0] = 1 - 2 * cx / width
    P[2, 1] = 2 * cy / height - 1
    P[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    P[2, 3] = -1.0
    P[3, 2] = - (2 * far_plane * near_plane) / (far_plane - near_plane)

    # print("near, far: ", near_plane, far_plane)
    # print("Intrinsic: ")
    # print(MTX)
    # print("Projection Matrix Transpose")
    # print(P) 

    return P.flatten()


class AR_render:
    
    def __init__(self, camera_matrix, dist_coefs, rvec, tvec, image):

        self.image = image        
        self.image_h, self.image_w = image.shape[:2]
        self.rvec, self.tvec = rvec, tvec

        self.initOpengl(self.image_w, self.image_h)
    
        self.cam_matrix,self.dist_coefs = camera_matrix, dist_coefs
        self.projectMatrix = intrinsic2Project(camera_matrix, self.image_w, self.image_h, 0.01, 100.0)
        
        # Model translate that you can adjust by key board 'w', 's', 'a', 'd'
        self.translate_x, self.translate_y, self.translate_z = 0, 0, 0

        self.frameCounter = 0

    #

    def initOpengl(self, width, height, pos_x = 800, pos_y = 500, window_name = 'ARR'):
        
        """[Init opengl configuration]
        
        Arguments:
            width {[int]} -- [width of opengl viewport]
            height {[int]} -- [height of opengl viewport]
        
        Keyword Arguments:
            pos_x {int} -- [X cordinate of viewport] (default: {500})
            pos_y {int} -- [Y cordinate of viewport] (default: {500})
            window_name {bytes} -- [Window name] (default: {b'Aruco Demo'})
        """
        
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(pos_x, pos_y)
        
        self.window_id = glutCreateWindow(window_name)
        glutDisplayFunc(self.draw_scene)
        glutIdleFunc(self.draw_scene)
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        
        # # Assign texture
        glEnable(GL_TEXTURE_2D)
        
        # Add listener
        glutKeyboardFunc(self.keyBoardListener)
        
        # Set ambient lighting
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5,0.5,0.5,1)) 
    #
 
 # https://github.com/hughesj919/PyAugmentedReality/blob/master/Augment.py
    def draw_scene(self):
        self.frameCounter += 1
        if 10:
            flippedImage = cv2.flip(self.image, 0)
            flippedImage[100:200, 300:500] = [200, 10, 10]
            glDisable(GL_DEPTH_TEST)
            glDrawPixels(flippedImage.shape[1], flippedImage.shape[0], GL_BGR, GL_UNSIGNED_BYTE, flippedImage.data)
            glEnable(GL_DEPTH_TEST)

        glViewport(0, 0, self.image_w, self.image_h)

        self.draw_objects() # draw the 3D objects.

        glutSwapBuffers()
     
 
    def draw_objects(self):

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        principalX = self.cam_matrix[0, 2]
        principalY = self.cam_matrix[1, 2]
        fx = self.cam_matrix[0, 0]
        fy = self.cam_matrix[1, 1]
        near = 1
        far = 400
        width = self.image_w
        height = self.image_h

        glFrustum(-principalX / fx, (width - principalX) / fy, (principalY - height) / fy, principalY / fy, near, far)

        # projectMatrix = intrinsic2Project(self.cam_matrix, self.image_w, self.image_h, 0.01, 100.0)
        # glMultMatrixf(projectMatrix)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        model_matrix = extrinsic2ModelView(self.rvec, self.tvec)
        glLoadMatrixf(model_matrix)
        # print(model_matrix)

        glDisable(GL_DEPTH_TEST)
        # draw an object here
        glLineWidth(7)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0.,0, 0)
        glVertex3f(5.,0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 5, 0)
        glColor3f(0.2, 0.2, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 5)
        glEnd()
        glColor3f(1, 1, 1)
        # --------------------            
            
        cv2.imshow("Frame", self.image)
        ch = cv2.waitKey(20)
        # print(int(ch))
        if ch == 32: exit()
        print(f"draw_objects({self.frameCounter}) done.")

    def keyBoardListener(self, key, x, y):
        """[Use key board to adjust model size and position]
        
        Arguments:
            key {[byte]} -- [key value]
            x {[x cordinate]} -- []
            y {[y cordinate]} -- []
        """
        key = key.decode('utf-8')
        if key == '=':
            self.model_scale += 0.01
        elif key == '-':
            self.model_scale -= 0.01
        elif key == 'x':
            self.translate_x -= 0.01
        elif key == 'X':
            self.translate_x += 0.01
        elif key == 'y':
            self.translate_y -= 0.01
        elif key == 'Y':
            self.translate_y += 0.01
        elif key == 'z':
            self.translate_z -= 0.01
        elif key == 'Z':
            self.translate_z += 0.01 
        elif key == '0':
            self.translate_x, self.translate_y, self.translate_z = 0, 0, 0
        
    def run(self):
        # Begin to render
        glutMainLoop()
  
