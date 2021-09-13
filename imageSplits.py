            #cv2 split
            #l, a, b = cv2.split(img)
            #st.image(l)
            #st.image(a)
            #st.image(b)

            #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            #cl = clahe.apply(l)
            #limg = cv2.merge((cl,a,b))
            #st.image(cl)
            #lab2rbg
            #lab =cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            #st.image(lab)

            #Blanqueo, mucho
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            #dilated = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
            #diff1 = 255 - cv2.subtract(dilated, img)

            #median = cv2.medianBlur(dilated, 15)
            #diff2 = 255 - cv2.subtract(median, img)

            #normed = cv2.normalize(diff2,None, 0, 255, cv2.NORM_MINMAX )

            #dst = np.hstack((img, normed))
            #res = np.hstack((img,dilated, diff1,  median, diff2, normed))

            #st.image(dst)
            #st.image(res)