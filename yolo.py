bounding_box_annotations={}
for name,image in images.items():
    plt.figure()
    bounding_box_annotations[name]=mask_to_annotation(image, type='bounding_box')
    for count,contours in enumerate(bounding_box_annotations[name]):

        x,y,w,h = cv2.boundingRect(contours)
        print(x)
        print(y)
        print(w)
        print(h)
        print()
#         box=bounding_box_annotations[name][y:y+h, x: x+w]
#         #Drawing rectangle for ROI of current line Contour
#         cv2.rectangle(box, (x,y), (x+w, y+h), (40, 100, 250), 2)

#         plt.imshow(box)
#         plt.axis('off')