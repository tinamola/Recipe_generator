import os
import numpy as np
import torch.nn.utils
import torch.optim as optim

from time import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger
from utils import early_stopping, ensure_dir
from optim import ScheduledOptimizer
from evaluator import Evaluator
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()

        # training settings
        self.DDP = config['DDP']
        self.epochs = config['epochs']
        self.learner = config['learner'].lower()
        self.learning_rate = config['learning_rate']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.grad_clip = config['grad_clip']
        self.plot = config['plot']
        if self.plot:
            self.writer = SummaryWriter(log_dir='./runs/{}'.format(config['filename']))

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = 1e9
        self.best_valid_result = None
        self.optimizer = self._build_optimizer()

        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = config['filename'] + '.pth'
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.generated_text_dir = config['generated_text_dir']
        ensure_dir(self.generated_text_dir)
        saved_text_file = self.config['filename'] + '.txt'
        self.saved_text_file = os.path.join(self.generated_text_dir, saved_text_file)

        self.evaluator = Evaluator()

        self.is_logger = not self.DDP

    def _build_optimizer(self):
        if self.learner == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner == 'schedule':
            return ScheduledOptimizer(optim.Adam(self.model.parameters(), lr=self.learning_rate), self.config)

    def _train_epoch(self, train_data):
        self.model.train()
        total_loss = 0.

        pbar = train_data
        if self.is_logger:
            pbar = tqdm(pbar)

        for data in pbar:
            self.optimizer.zero_grad()
            loss = self.model(data)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        train_loss = total_loss / len(train_data)
        return train_loss

    def _valid_epoch(self, valid_data):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0.

            for data in valid_data:
                loss = self.model(data)
                total_loss += loss.item()

            valid_loss = total_loss / len(valid_data)
            ppl = np.exp(valid_loss)
        return valid_loss, ppl

    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.is_logger:
            self.logger.info('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

    def _save_generated_text(self, generated_corpus):
        with open("generated.txt", 'w') as fin:
            for tokens in generated_corpus:
                fin.write(' '.join(tokens) + '\n')

    def fit(self, train_data, valid_data, saved=True):
        if self.start_epoch >= self.epochs or self.epochs <= 0:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            train_loss = self._train_epoch(train_data)
            training_end_time = time()

            train_loss_output = "epoch %d training [time: %.2fs, train_loss: %.4f]" % \
                                (epoch_idx, training_end_time - training_start_time, train_loss)
            if self.is_logger:
                self.logger.info(train_loss_output)

            if self.plot:
                self.writer.add_scalar('loss', train_loss, epoch_idx)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step, max_step=self.stopping_step
                )
                valid_end_time = time()

                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_loss: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid ppl: {}'.format(valid_result)
                if self.is_logger:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if self.is_logger:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result
                if stop_flag:
                    stop_output = ('Finished training, best eval result in epoch %d' %
                                   (epoch_idx - self.cur_step * self.eval_step))
                    if self.is_logger:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, model_file=None):
        if model_file:
            checkpoint_file = model_file
        else:
            checkpoint_file = self.saved_model_file
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
        if self.is_logger:
            self.logger.info(message_output)

        self.model.eval()

        generated_corpus1 = []
        with torch.no_grad():
            for data in tqdm(eval_data):
                generated = self.model.generate(data)
                generated_corpus1.extend(generated)

            # print(generated_corpus)

        # self._save_generated_text(generated_corpus)
        reference_corpus1 = eval_data.get_reference()
        reference_input = eval_data.get_input() #[["as",";","sdf"],["fsd",";",";"]]
        reference_input1 = [[value for value in temp if value != ';'] for temp in reference_input]
        #print(len(reference_input1))
        # with open("reference.txt", 'w') as fin:
        #     for tokens in reference_corpus:
        #         fin.write(' '.join(tokens) + '\n')
        #result = self.evaluator.evaluate(generated_corpus, reference_corpus)
        generated_corpus = [" ".join(generated_sentence) for generated_sentence in generated_corpus1]
        reference_corpus = [" ".join(reference_sentence) for reference_sentence in reference_corpus1]

        # generated_corpus, reference_corpus = list(map(list,zip(*((input_d, output_d) for input_d, output_d in zip(generated_corpus, reference_corpus) if len(input_d)!=0 and len(output_d)!=0))))
        print(reference_input1[5])
        print(reference_corpus[5])
        total_score_bleu1,total_score_bleu2,total_score_bleu3,total_score_meteor,coverage=0,0,0,0,0

        for i in range(len(reference_corpus)):
            reference1=[reference_corpus[i]]
            candidate1=generated_corpus[i]
            reference2=[reference_corpus1[i]]
            candidate2=generated_corpus1[i]
            inputreference=reference_input1[i]
            score1 = sentence_bleu(reference1, candidate1,[1,0,0,0])
            score2 = sentence_bleu(reference1, candidate1,[0,1,0,0])
            score3 = sentence_bleu(reference1, candidate1,[0,0,0,1])
            score4 = meteor_score(reference2, candidate2)
            total_score_bleu1+=score1
            total_score_bleu2+=score2
            total_score_bleu3+=score3
            total_score_meteor+=score4
            coverage+=(len(inputreference) - len(set(inputreference) - set(candidate2)))/len(inputreference)
        return "bleu1: " + str(round(total_score_bleu1/len(reference_corpus),4))+ " bleu2: " + str(round(total_score_bleu2/len(reference_corpus),4))+ " bleu4: " + str(round(total_score_bleu3/len(reference_corpus),4))+" meteor: " + str(round(total_score_meteor/len(reference_corpus),4))+" coverage: " + str(round(coverage/len(reference_corpus),4))

