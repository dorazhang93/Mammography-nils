import cv2
import numpy as np

def sigmoid_project(arr, c,w, alpha = 4):
    arr = 255/(1 + np.exp(alpha*(c-arr)/w))
    return arr
def linear_project(arr, c,w):
    arr = ((arr-c)/w+0.5)*255
    return np.clip(arr, 0, 255)
def scale_0_255(img):
    img=((img-img.min())/(img.max()-img.min()))*255
    return img

class DMImagePreprocessor(object):
    '''Class for preprocessing images in the DM challenge
    '''

    def __init__(self):
        '''Constructor for DMImagePreprocessor
        '''
        pass

    def select_largest_obj(self, img_bin, lab_val=255, fill_holes=False,
                           smooth_boundary=False, kernel_size=15):
        '''Select the largest object from a binary image and optionally
        fill holes inside it and smooth its boundary.
        Args:
            img_bin (2D array): 2D numpy array of binary image.
            lab_val ([int]): integer value used for the label of the largest
                    object. Default is 255.
            fill_holes ([boolean]): whether fill the holes inside the largest
                    object or not. Default is false.
            smooth_boundary ([boolean]): whether smooth the boundary of the
                    largest object using morphological opening or not. Default
                    is false.
            kernel_size ([int]): the size of the kernel used for morphological
                    operation. Default is 15.
        Returns:
            a binary image as a mask for the largest object.
        '''
        n_labels, img_labeled, lab_stats, _ = \
            cv2.connectedComponentsWithStats(img_bin, connectivity=8,
                                             ltype=cv2.CV_32S)
        largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
        largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
        largest_mask[img_labeled == largest_obj_lab] = lab_val
        # import pdb; pdb.set_trace()
        if fill_holes:
            bkg_locs = np.where(img_labeled == 0)
            bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
            img_floodfill = largest_mask.copy()
            h_, w_ = largest_mask.shape
            mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
            cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed,
                          newVal=lab_val)
            holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
            largest_mask = largest_mask + holes_mask
        if smooth_boundary:
            kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN,
                                            kernel_)

        return largest_mask

    @staticmethod
    # Make all image have the same resolution (GE: 0.1) -->
    def resolution(image, ips):  # ips = Imager Pixel Spacing
        # Smaller the detector element size, the higher is the resolution.
        wantedResolution = 0.09409090909091  # mm/pixel --> 10 pixels per mm
        # If we have 0.07 mm/pixel --> 14 pixels per mm
        scalingFactor = ips / wantedResolution  # scalingFactor < 1 --> worse resolution
        width = int(image.shape[1] * scalingFactor)
        height = int(image.shape[0] * scalingFactor)
        dim = (width, height)  # desired size for the output image
        return cv2.resize(image, dim,
                          interpolation=cv2.INTER_LINEAR)  # One thing to keep in mind while using the cv2.resize() function is that the tuple passed for determining the size of the new image follows the order (width, height) unlike as expected (height, width).

    @staticmethod
    def max_pix_val(dtype):
        if dtype == np.dtype('uint8'):
            maxval = 2 ** 8 - 1
        elif dtype == np.dtype('uint16'):
            maxval = 2 ** 16 - 1
        else:
            raise Exception('Unknown dtype found in input image array')
        return maxval

    @staticmethod
    def relax_bbox( img_size, bbox):
        thre=0.01
        img_h, img_w =img_size
        x, y, w, h = bbox
        def relax_width(r):
            x_s = int(np.max((0, int(x-w*(r-1)/2))))
            x_e = int(np.min((img_w - 1, int(w * r + x - w * (r - 1) / 2))))
            return x_s, x_e

        def reduce_top(r):
            y_s = int(y + h * r)
            h_new = int(h-h*r)
            return y_s, h_new

        def reduce_bottom(r,hh):
            h_new = int(hh-hh*r)
            return h_new

        if h/w <2:
            x_s, x_e =relax_width(1.4)
            return x_s, y, x_e-x_s, h
        if h/w>=2 and h/w<3:
            x_s, x_e = relax_width(1.4)
            if y>thre*img_h:
                return x_s, y, x_e - x_s, h
            else:
                y, h = reduce_top(0.15)
                return x_s, y, x_e - x_s, h
        if h/w>=3:
            x_s, x_e = relax_width(1.5)
            if y<thre*img_h:
                y, h = reduce_top(0.2)
            if (y+h)>(img_h*(1-thre)):
                h = reduce_bottom(0.1,h)
            return x_s, y, x_e-x_s, h

    def suppress_artifacts(self, img, global_threshold=.05, kernel_size=15):
        '''Mask artifacts from an input image
        Artifacts refer to textual markings and other small objects that are
        not related to the breast region.

        Args:
            img (2D array): input image as a numpy 2D array.
            global_threshold ([int]): a global threshold as a cutoff for low
                    intensities for image binarization. Default is 18.
            kernel_size ([int]): kernel size for morphological operations.
                    Default is 15.
        Returns:
            a tuple of (output_image, breast_mask). Both are 2D numpy arrays.
        '''
        maxval = 255
        if global_threshold < 1.:
            low_th = int(img.max() * global_threshold)
        else:
            low_th = int(global_threshold)
        _, img_bin = cv2.threshold(img, low_th, maxval=maxval,
                                   type=cv2.THRESH_BINARY)
        breast_mask = self.select_largest_obj(img_bin, lab_val=maxval,
                                              fill_holes=True,
                                              smooth_boundary=True,
                                              kernel_size=kernel_size)
        img_suppr = cv2.bitwise_and(img, breast_mask)

        return (img_suppr, breast_mask)

    @classmethod
    def segment_breast(self, img, low_int_threshold=0.05, erode_kernel=10, n_erode = 10):
        '''Perform breast segmentation
        Args:
            low_int_threshold([float or int]): Low intensity threshold to
                    filter out background. It can be a fraction of the max
                    intensity value or an integer intensity value.
            crop ([bool]): Whether or not to crop the image.
        Returns:
            An image of the segmented breast.
        NOTES: the low_int_threshold is applied to an image of dtype 'uint8',
            which has a max value of 255.
        '''
        # Create img for thresholding and contours.
        img_8u = img
        if low_int_threshold < 1.:
            low_th = int(img_8u.max() * low_int_threshold)
        else:
            low_th = int(low_int_threshold)
        _, img_bin = cv2.threshold(
            img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
        img_bin = cv2.erode(img_bin,kernel=np.ones((erode_kernel, erode_kernel), dtype=np.uint8),iterations=n_erode)
        img_bin = cv2.dilate(img_bin,kernel=np.ones((erode_kernel, erode_kernel), dtype=np.uint8),iterations=n_erode)
        contours, _ = cv2.findContours(
            img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [cv2.contourArea(cont) for cont in contours]
        idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
        x, y, w, h = cv2.boundingRect(contours[idx])
        return self.relax_bbox(img.shape, (x, y, w, h))


    def project_255(self, img, window=None, mode="Sigmoid"):
        ''' Project uint16 pixel value to float value ranged from 0 - 255'''
        max = img.max()
        min = img.min()
        if window is None:
            c = int((max+min)/2)
            w = int(max-min+1)
        else:
            try:
                num = len(window[0])
                # print("using dicom center & width index", num)
                c = int(window[0][num-1])
                w = int(window[1][num - 1])
            except:
                # print("using dicom center & width index 0")
                c = int(window[0])
                w = int(window[1])
        if mode == "Sigmoid":
            img = sigmoid_project(img, c,w)
            img = scale_0_255(img)
        elif mode == "Linear":
            img = linear_project(img,  c,w)
        else:
            raise ValueError("Invalid projection mode, "+mode)
        return img

    def invert(self,img):
        return img.max()-img

    def process(self, img, project=True, project_mode="Sigmoid",
                artif_suppression=True, low_int_threshold=.05, kernel_size=15,
                segment_breast=True, segment_low_int_threshold=.05, segment_erode_kernel=10,segment_n_erode=10):
        '''Perform multi-stage preprocessing on the input image
        Args:
            project: if project from dicom pixcel space to 0-255 with the given project_mode
            low_int_threshold ([int]): cutoff used in artifacts suppression.
        Returns:
            a tuple of (processed_image, bbox). If
            segment breast was not called.
        '''
        pixcelspacing = None
        if not isinstance(img,np.ndarray):
            try:
                if img.data_element("PhotometricInterpretation").value == "MONOCHROME1":
                    img_proc = self.invert(img.pixel_array.astype(int))
                    print("Inverted MONOCHROME1")
                elif img.data_element("PhotometricInterpretation").value == "MONOCHROME2":
                    img_proc = (img.pixel_array).astype(int)
                else:
                    print(f"Invalid PhotometricInterpretation {img.data_element('PhotometricInterpretation').value}")
                    return (None, None)
            except:
                return (None,None)
            window = (img.data_element('WindowCenter').value,img.data_element('WindowWidth').value) if \
                ((0x0028, 0x1050) in img and (0x0028, 0x1051) in img) else None
            pixcelspacing = img.data_element('ImagerPixelSpacing').value[0] if (0x0018, 0x1164) in img else None
            if project:
                img_proc = self.project_255(img_proc, window= window, mode=project_mode)

        else:
            img_proc = img[:,:,0]

        if pixcelspacing is not None and pixcelspacing != 0.09409090909091:
            self.resolution(img_proc, pixcelspacing)

        if artif_suppression:
            try:
                img_proc, _ = self.suppress_artifacts(
                    img_proc.astype("uint8"), global_threshold=low_int_threshold,
                    kernel_size=kernel_size)
            except:
                return None, None

        if segment_breast:
            try:
                bbox = self.segment_breast(img_proc.astype("uint8"),
                                           low_int_threshold=segment_low_int_threshold,
                                           erode_kernel=segment_erode_kernel,
                                           n_erode=segment_n_erode)
            except:
                return None, None
        else:
            bbox = None

        return (img_proc.astype("uint8"), bbox)