import os
import cv2
import numpy as np
import subprocess
import h5py
import soundfile as sf
import logging
import traceback
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from talkingface.utils import face_detection
import traceback
import librosa
import librosa.filters
import pickle
from scipy import signal
from scipy.io import wavfile


class lrs2Preprocess:
    def __init__(self, config):
        self.config = config
        self.fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
                                                device=f'cuda:{id}') for id in range(config['ngpu'])]
        self.template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

    def process_video_file(self, vfile, gpu_id):
        video_stream = cv2.VideoCapture(vfile)
        
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)
        
        vidname = os.path.basename(vfile).split('.')[0]
        dirname = vfile.split('/')[-2]

        fulldir = os.path.join(self.config['preprocessed_root'], dirname, vidname)
        os.makedirs(fulldir, exist_ok=True)

        batches = [frames[i:i + self.config['preprocess_batch_size']] for i in range(0, len(frames), self.config['preprocess_batch_size'])]

        i = -1
        for fb in batches:
            preds = self.fa[gpu_id].get_detections_for_batch(np.asarray(fb))

            for j, f in enumerate(preds):
                i += 1
                if f is None:
                    continue

                x1, y1, x2, y2 = f
                cv2.imwrite(os.path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])


    def process_audio_file(self, vfile):
        vidname = os.path.basename(vfile).split('.')[0]
        dirname = vfile.split('/')[-2]

        fulldir = os.path.join(self.config['preprocessed_root'], dirname, vidname)
        os.makedirs(fulldir, exist_ok=True)

        wavpath = os.path.join(fulldir, 'audio.wav')

        command =self.template.format(vfile, wavpath)
        subprocess.call(command, shell=True)

    def mp_handler(self, job):
        vfile, gpu_id = job
        try:
            self.process_video_file(vfile, gpu_id)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()

    def run(self):
        print(f'Started processing for {self.config["data_root"]} with {self.config["ngpu"]} GPUs')
        
        filelist = glob(os.path.join(self.config["data_root"], '*/*.mp4'))

        # jobs = [(vfile, i % self.config["ngpu"]) for i, vfile in enumerate(filelist)]
        # with ThreadPoolExecutor(self.config["ngpu"]) as p:
        #     futures = [p.submit(self.mp_handler, j) for j in jobs]
        #     _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

        print('Dumping audios...')
        for vfile in tqdm(filelist):
            try:
                self.process_audio_file(vfile)
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
                continue

class arcticPreprocess:
    def __init__(self, config):
        self.config = config
        
    def walk_files(self, root, extension):
        for path, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(extension):
                    yield os.path.join(path, file)

    def logmelfilterbank(self,
                         audio,
                         sampling_rate,
                         fft_size=1024,
                         hop_size=256,
                         win_length=None,
                         window="hann",
                         num_mels=80,
                         fmin=None,
                         fmax=None,
                         eps=1e-10,
                         ):
        """Compute log-Mel filterbank feature.
        Args:
            audio (ndarray): Audio signal (T,).
            sampling_rate (int): Sampling rate.
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length. If set to None, it will be the same as fft_size.
            window (str): Window function type.
            num_mels (int): Number of mel basis.
            fmin (int): Minimum frequency in mel basis calculation.
            fmax (int): Maximum frequency in mel basis calculation.
            eps (float): Epsilon value to avoid inf in log calculation.
        Returns:
            ndarray: Log Mel filterbank feature (#frames, num_mels).
        """
        # get amplitude spectrogram
        x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                              win_length=win_length, window=window, pad_mode="reflect")
        spc = np.abs(x_stft).T  # (#frames, #bins)

        # get mel basis
        fmin = 0 if fmin is None else fmin
        fmax = sampling_rate / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

        return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    def extract_melspec(self, src_filepath, dst_filepath):
        try:
            trim_silence = self.config['trim_silence']
            top_db = self.config['top_db']
            flen = self.config['flen']
            fshift = self.config['fshift']
            fmin = self.config['fmin']
            fmax = self.config['fmax']
            num_mels = self.config['num_mels']
            fs = self.config['fs']

            audio, fs_ = sf.read(src_filepath)
            if trim_silence:
                #print('trimming.')
                audio, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=2048, hop_length=512)
            if fs != fs_:
                #print('resampling.')
                audio = librosa.resample(audio, fs_, fs)
            melspec_raw = self.logmelfilterbank(audio,fs, fft_size=flen,hop_size=fshift,
                                            fmin=fmin, fmax=fmax, num_mels=num_mels)
            melspec_raw = melspec_raw.astype(np.float32)
            melspec_raw = melspec_raw.T # n_mels x n_frame

            if not os.path.exists(os.path.dirname(dst_filepath)):
                os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
            with h5py.File(dst_filepath, "w") as f:
                f.create_dataset("melspec", data=melspec_raw)
                
            logging.info(f"{dst_filepath}...[{melspec_raw.shape}].")

        except Exception as e:  # 捕获所有的异常
            print(f"{dst_filepath}...failed.")
            print(f"Exception occurred: {e}")
            traceback.print_exc()  # 打印异常的堆栈跟踪
            
    def read_melspec(self, filepath):
        with h5py.File(filepath, "r") as f:
            melspec = f["melspec"][()]  # n_mels x n_frame
        #import pdb;pdb.set_trace() # Breakpoint
        return melspec

    def compute_statistics(self, src, stat_filepath):
        melspec_scaler = StandardScaler()

        filepath_list = list(self.walk_files(src, '.h5'))
        for filepath in tqdm(filepath_list):
            melspec = self.read_melspec(filepath)
            #import pdb;pdb.set_trace() # Breakpoint
            melspec_scaler.partial_fit(melspec.T)

        with open(stat_filepath, mode='wb') as f:
            pickle.dump(melspec_scaler, f)

    def melspec_transform(self, melspec, scaler):
        # melspec.shape: (n_freq, n_time)
        # scaler.transform assumes the first axis to be the time axis
        melspec = scaler.transform(melspec.T)
        #import pdb;pdb.set_trace() # Breakpoint
        melspec = melspec.T
        return melspec

    def normalize_features(self, src_filepath, dst_filepath, melspec_transform, melspec_scaler):
        try:
            with h5py.File(src_filepath, "r") as f:
                melspec = f["melspec"][()]
            melspec = self.melspec_transform(melspec, melspec_scaler)

            if not os.path.exists(os.path.dirname(dst_filepath)):
                os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
            with h5py.File(dst_filepath, "w") as f:
                f.create_dataset("melspec", data=melspec)

            #logging.info(f"{dst_filepath}...[{melspec.shape}].")
            return melspec.shape

        except Exception as e:  # 捕获所有异常
            logging.info(f"{dst_filepath}...failed.")
            logging.info(f"Exception occurred: {e}")
            logging.info("Traceback details:")
            traceback_details = traceback.format_exc()
            logging.info(traceback_details)
    
    def run(self):
        src1 = self.config['src1']
        dst1 = self.config['dst1']
        ext1 = self.config['ext1']
        stat_filepath = self.config['stat']
        src2 = self.config['src2']
        dst2 = self.config['dst2']
        ext2 = self.config['ext2']
        fargs_list = [
            [
                f,
                f.replace(src1, dst1).replace(ext1, ".h5"),
            ]
            for f in self.walk_files(src1, ext1)
        ]
        
        results = []
        for f in tqdm(fargs_list):
            result = self.extract_melspec(*f)
            results.append(result)
        
        self.compute_statistics(src2, stat_filepath)
        
        melspec_scaler = StandardScaler()
        if os.path.exists(stat_filepath):
            with open(stat_filepath, mode='rb') as f:
                melspec_scaler = pickle.load(f)
            print('Loaded mel-spectrogram statistics successfully.')
        else:
            print('Stat file not found.')
        root = src2
        fargs_list = [
            [
                f,
                f.replace(src2, dst2),
                lambda x: self.melspec_transform(x, melspec_scaler),
            ]
            for f in self.walk_files(root, ext2)
        ]
        for f in tqdm(fargs_list):
            # 解包 f，传入 scaler
            src, dst, transform_func = f
            result = self.normalize_features(src, dst, transform_func, melspec_scaler)
            results.append(result)
