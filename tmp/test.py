import argparse
import os
import yaml
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models import encoders

from data.datasets import idrnd
from data.transform import Transforms

from utils.storage import load_weights


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--path-images-csv', type=str, required=True)
    parser.add_argument('--path-test-dir', type=str, required=True)
    parser.add_argument('--path-submission-csv', type=str, required=True)
    args = parser.parse_args()

    # prepare image paths
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    test_dataset_paths = pd.read_csv(args.path_images_csv)
    path_test_dir = args.path_test_dir

    paths = [
        {
            'id': row.id,
            'frame': row.frame,
            'path': os.path.join(path_test_dir, row.path)
        } for _, row in test_dataset_paths.iterrows() if int(row.frame) in config['frames']]
    test_df = pd.DataFrame(paths)
    
    test_loader = DataLoader(idrnd.TestAntispoofDataset(
        test_df, Transforms(input_size=config['input_size'], train=False), config['tta']),
                             batch_size=config['batch_size'],
                             num_workers=config['num_workers'],
                             shuffle=False)

    model = getattr(encoders, config['encoder'])(device=device,
                                                 out_features=1,
                                                 pretrained=False)
    load_weights(model, config['prefix'], 'model', 'best')
    model.eval()

    samples, frames, probabilities = [], [], []

    with torch.no_grad():
        for batch, video, frame in test_loader:
            batch = batch.to(device)
            probability = torch.sigmoid(model(batch).view(-1))

            samples.extend(video)
            frames.extend(frame.numpy())
            probabilities.extend(probability.cpu().numpy())

    # save
    predictions = pd.DataFrame.from_dict({
        'id': samples,
        'frame': frames,
        'probability': probabilities})

    predictions = predictions.groupby('id').probability.mean().reset_index()
    predictions['prediction'] = predictions.probability
    predictions[['id', 'prediction']].to_csv(
        args.path_submission_csv, index=False)