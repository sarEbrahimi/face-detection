import cv2

# compare histogram
# main_hist is histogram which image the user enter

def comp_his(hist,main_hist):
    compare_hist = cv2.compareHist(hist , main_hist , cv2.HISTCMP_CORREL)
    return compare_hist