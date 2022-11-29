import os
import cv2
import argparse
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(
            prog = 'Demo',
            description = 'Demo multi-camera tracking',
            epilog = 'Text at the bottom of help')

    parser.add_argument('-d', '--data', default='./AIC21_Track2_ReID')              # path to data directory
    parser.add_argument('-p', '--pred', default='./sample-predictions.txt')         # path to prediction file
    parser.add_argument('-o', '--out', default='./pred_visual_results')             # path to output folder
    
    return parser.parse_args()


def get_data(data_path):
    """
        Return the list of query data and the list of test data
        Argument:
            data_path (str) : path to data directory
    """
    query = os.listdir(os.path.join(data_path, 'image_query'))
    test = os.listdir(os.path.join(data_path, 'image_test'))
    return query, test

def get_matches(data_path):
    """
        Return dictionary of query data matched to test data
        Argument:
            data_path (str) : path to prediction file
    """
    matches = {}
    with open(data_path, "r") as f:
        lines = f.readlines()
        if lines[-1] == "":
            lines = lines[:-1]

        for query_id, line in enumerate(lines):
            test_ids = line[:-1].split(" ")
            matches[query_id+1] = test_ids

    return matches

def read_image(image_path, size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    return image

def make_border(image, color):
    image = cv2.copyMakeBorder(image,10,10,10,10, cv2.BORDER_CONSTANT,value=color)
    return image

if __name__ == '__main__':
    args = get_arguments()

    data_path       = args.data
    prediction_file = args.pred
    output_folder   = args.out
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    query, test = get_data(data_path)

    matches = get_matches(prediction_file)

    batch_query_id = 1
    batch_size = 5
    while True:
        final_frame = None

        for id in range(batch_size):
            query_id = batch_query_id + id
            query_path = os.path.join(data_path, 'image_query', str(query_id).zfill(6) + '.jpg')
            if not os.path.exists(query_path):
                break

            test_ids = matches[query_id]
            test_paths = [os.path.join(data_path, 'image_test', str(test_id).zfill(6) + '.jpg')
                        for test_id in test_ids]

            query_image = read_image(query_path, size=(150, 150))
            query_image = make_border(query_image, color=(0,0,255))
            tests = [read_image(test_path, size=(150, 150)) for test_path in test_paths[:8]] # read first 8 images
            tests = [make_border(test, color=[255,255,255]) for test in tests]
            test_image = cv2.hconcat(tests)

            query_final_frame = cv2.hconcat([query_image, test_image])

            if final_frame is None:
                final_frame = query_final_frame
            else:
                final_frame = cv2.vconcat([final_frame, query_final_frame])
        
        if final_frame is not None:
            out_name = 'Batch ' + str(batch_query_id // batch_size + 1) + ': query (left) - model predictions (right)'
            out_path = os.path.join(output_folder, out_name+'.jpg')
            cv2.imwrite(out_path, final_frame) # write outputs
            batch_query_id += batch_size
        else:
            break

        # cv2.imshow(out_name, final_frame)
        # __key__ = cv2.waitKey(1)
        # if __key__ == ord('q'):
        #     break
        # elif __key__ == ord('n'):
        #     cv2.destroyAllWindows()

    cv2.destroyAllWindows()

