#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''
import numpy as np
import cv2 as cv
import cv2 

# local modules
from common import splitfn

# built-in modules
import os


class CalibAR :
    def __init__(self, argv=[]):
        self.imgs = []
        self.imgfnames = []
        self.main(argv)

    def main(self, argv):
        import getopt
        from glob import glob

        args, img_mask = getopt.getopt(argv[1:], '', ['debug=', 'square_size=', 'threads='])
        args = dict(args)
        args.setdefault('--debug', './output/')
        args.setdefault('--square_size', 1.0)
        args.setdefault('--threads', 1)
        if not img_mask:
            img_mask = './data/left??.jpg'  # default
        else:
            img_mask = img_mask[0]

        img_names = glob(img_mask)
        img_names.sort()
        debug_dir = args.get('--debug')
        if debug_dir and not os.path.isdir(debug_dir):
            os.mkdir(debug_dir)
        square_size = float(args.get('--square_size'))

        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        obj_points = []
        img_points = []
        h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results

        def processImage(fn):
            print('processing %s... ' % fn)
            img = cv.imread(fn, 0)

            self.imgs.append(img) 
            self.imgfnames.append(fn)

            if img is None:
                print("Failed to load", fn)
                return None

            assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
            found, corners = cv.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            if debug_dir:
                vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                cv.drawChessboardCorners(vis, pattern_size, corners, found)
                _path, name, _ext = splitfn(fn)
                outfile = os.path.join(debug_dir, name + '_chess.png')
                cv.imwrite(outfile, vis)

            if not found:
                print('chessboard not found')
                return None

            print('           %s... OK' % fn)
            return (corners.reshape(-1, 2), pattern_points)

        threads_num = int(args.get('--threads'))
        if threads_num <= 1:
            chessboards = [processImage(fn) for fn in img_names]
        else:
            print("Run with %d threads..." % threads_num)
            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(threads_num)
            chessboards = pool.map(processImage, img_names)

        chessboards = [x for x in chessboards if x is not None]
        for (corners, pattern_points) in chessboards:
            img_points.append(corners)
            obj_points.append(pattern_points)

        # calculate camera distortion
        rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.rvecs = _rvecs
        self.tvecs = _tvecs


        print("\nRMS:", rms)
        print("camera matrix:\n", camera_matrix)
        print("distortion coefficients: ", dist_coefs.ravel())

        print("----- R, t ------")
        for i, (rv, tv) in enumerate(zip(_rvecs, _tvecs)):
            R, Jacob = cv.Rodrigues(rv)
            print(f"----- cam [{i}] -----")
            print(rv.T)
            print(tv.T)
            print(R)
            break

        # undistort the image with the calibration
        print('')
        for fn in img_names if debug_dir else []:
            _path, name, _ext = splitfn(fn)
            img_found = os.path.join(debug_dir, name + '_chess.png')
            outfile = os.path.join(debug_dir, name + '_undistorted.png')

            img = cv.imread(img_found)
            if img is None:
                continue

            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

            dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

            # crop and save the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            print('Undistorted image written to: %s' % outfile)
            cv.imwrite(outfile, dst)

        print('Done')

# class CalibAR 

# import ARR  # AR Rendering

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
        self.tz = 0.
        self.tstep = 0.1

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
        if self.tz > 0 or self.tz < -10: self.tstep *= -1
        self.tz += self.tstep;

        img = self.image.copy()
        cv2.putText(img, f"rotX: {self.frameCounter % 360}", (450, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img, f"Tz: {self.tz:5.2}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 155, 100), 2)

        if 10:
            flippedImage = cv2.flip(img, 0)  # upside down because of the order of mem reading of opengl
        else:
            flippedImage = img 
        
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

        if 10:
            # due to OpenGL's coordinate system, we need to swap bottom and top
            glFrustum(-principalX / fx, (width - principalX) / fx, (principalY - height) / fy, principalY / fy, near, far)
        else:
            glFrustum(-principalX / fx, (width - principalX) / fx, -principalY / fy, (height - principalY) / fy, near, far)

        # projectMatrix = intrinsic2Project(self.cam_matrix, self.image_w, self.image_h, 0.01, 100.0)
        # glMultMatrixf(projectMatrix)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        model_matrix = extrinsic2ModelView(self.rvec, self.tvec)
        glLoadMatrixf(model_matrix)
        # print(model_matrix)

        glPushMatrix() 
        if 1:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_DEPTH_TEST)
            glClear(GL_DEPTH_BUFFER_BIT)
            glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
            glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
            glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
            glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
            glRotatef(-90, 1, 0, 0)
            glutSolidTeapot(2)
        glPopMatrix()

        glPushMatrix() # blue teapot
        if 1:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_DEPTH_TEST)
            glClear(GL_DEPTH_BUFFER_BIT)
            glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
            glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.1,0.0,0.7,0.0])
            glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
            glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
            glTranslatef(8, 0, 0)
            glRotatef(-10, 1, 0, 0)
            glutSolidTeapot(2)
        glPopMatrix()

        glPushMatrix() 
        if 1:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_DEPTH_TEST)
            glClear(GL_DEPTH_BUFFER_BIT)
            glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
            glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.0,0.9,0.0,0.0])
            glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
            glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)

            glTranslatef(0, 5, self.tz)
            glRotatef(-90, 1, 0, 0)
            glutSolidTeapot(2)
        glPopMatrix()

        # glutSolidCone(GLdouble base_radius, GLdouble height, GLint slices, GLint stacks);
        # glutSolidCone and glutWireCone render a solid or wireframe cone respectively oriented along the Z axis. 
        # The base of the cone is placed at Z = 0, and the top at Z = height. 
        # The cone is subdivided around the Z axis into slices, and along the Z axis into stacks.
        glPushMatrix() 
        if 1:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_DEPTH_TEST)
            glClear(GL_DEPTH_BUFFER_BIT)
            glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
            glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
            glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)

            glTranslatef(8, 5, 0)
            glRotatef(self.frameCounter % 360, 1, 0, 0)
            glPushMatrix() # x-axis
            glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.9,0.1,0.0,0.0])
            glRotatef(90, 0, 1, 0)
            glutSolidCone(0.1, 3, 32, 100)
            glPopMatrix()
            glPushMatrix() # y-axis
            glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.1,0.9,0.0,0.0])
            glRotatef(-90, 1, 0, 0)
            glutSolidCone(0.1, 3, 32, 100)
            glPopMatrix()
        glPopMatrix()


        # glDisable(GL_DEPTH_TEST)
        # draw an object here
        glLineWidth(1)
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
        glPushMatrix()
        glTranslatef(5, 4, 0)
        glRotatef(self.frameCounter % 360, 1, 0, 0)
        glLineWidth(2)
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
        glPopMatrix()
        # --------------------            
                


        cv2.imshow("Frame", self.image)
        ch = cv2.waitKey(20)
        # print(int(ch))
        if ch == 32: exit()
        # print(f"draw_objects({self.frameCounter}) done.")

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
  




def GLrendering(car, image):
    ar_instance = AR_render(car.camera_matrix, car.dist_coefs, car.rvecs[0], car.tvecs[0], image)
    ar_instance.run()

if __name__ == '__main__':
    print(__doc__)
    car = CalibAR()

    print(car.imgfnames[0])
    print(car.camera_matrix)
    print("rv: ", car.rvecs[0].T)
    print("tv: ", car.tvecs[0].T)
    print(cv.Rodrigues(car.rvecs[0])[0])

    view = car.imgs[0].copy()  # gray
    view = cv.cvtColor(view, cv2.COLOR_GRAY2BGR)
    # openCV rendering
    if 0:
        view = cv.drawFrameAxes(view, car.camera_matrix, car.dist_coefs, car.rvecs[0], car.tvecs[0], 5)
        cv.imshow(f"{car.imgfnames[0]}", view)
        cv.waitKey()
        cv.destroyAllWindows()
    #
    else:
        GLrendering(car, view)
    #