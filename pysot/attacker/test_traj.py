from math import sin, cos, pi
import numpy as np
import cv2

def target_traj_gen_custom(init_rect,vid_h,vid_w,vid_l, type=1):
    '''
    type 1 : ellipse
    type 2 : rectangle
    type 3 : triangle
    '''
    target_traj = []
    pos = []
    pos.append(init_rect[0]+init_rect[2]/2)
    pos.append(init_rect[1]+init_rect[3]/2)
    target_traj.append(pos)

    # initial start_point , shape of traj
    def ellipse(t,a):
        return a*t*cos(t),a*t*sin(t)
    
    def rectangle(t,a):
        if t < 0.5:
            return t*a, 0
        elif t>= 0.5 and t<1:
            return 0.5*a, -(t-0.5)*a
        elif t>=1 and t<2:
            return -(t-1)*a + 0.5*a, -0.5*a
        elif t>=2 and t<3:
            return -0.5*a,(t-2)*a - 0.5*a
        else:
            return (t-3)*a -0.5*a,0.5*a 

        return 0,0

    def triangle(t,r):
        if t<1:
            return 0.5*r -  r*t*cos(pi/3), -r*t*sin(pi/3),
        elif t<2:
            return -(t-1)*r*cos(pi/3),  (t-2)*r*sin(pi/3)
        else :
            return  (t-2.5)*r , 0

    def line(t,r, theta):
        return t*r*cos(theta), t*r*sin(theta)

    r = min(vid_w,vid_h)

    for i in range(0,vid_l-1):
        tpos = []
        if type == 1:
            t = 4*pi*i/vid_l
            x,y = ellipse(t,r/(pi*8))
        if type == 2:
            t = 4.*i/vid_l
            x,y = rectangle(t,r/2)

        if type == 3:
            t = 3.*i/vid_l
            x,y = triangle(t,r/2)

        if type == 4:
            t = 1.*i/vid_l
            x, y = line(t,r, pi*0.25)
        
        tpos.append(np.clip(x + init_rect[0],0,vid_w-1))
        tpos.append(np.clip(y + init_rect[1],0,vid_h-1))
        target_traj.append(tpos)
    
    return target_traj

def visualization(target_traj,img):
    for pos in target_traj:
        cv2.circle(img, (int(pos[0]),int(pos[1])), 2, (255,255,255),-1)

if __name__ == '__main__':
    init_rect = [160,120,320,240]
    img = np.zeros([240,320,3],dtype=np.uint8)
    target_traj = target_traj_gen_custom(init_rect,240,320,200,type=4)
    visualization(target_traj,img)
    cv2.imshow('test',img)
    cv2.waitKey(0)