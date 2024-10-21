import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss
from pretrain.gather_layer import GatherLayer
from pytorch_lightning.utilities import rank_zero_only


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, config):
        super().__init__()
        self.model_points = model_points[0]
        if config['momentum_k'] == 0.0:
            self.model_points_k = self.model_points
        else:
            self.model_points_k = model_points[1]
        self.model_images = model_images
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.cycle_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.epoch = 0
        self.current_step = 0
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.working_dir = config["working_dir"]
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

        self._xent_targets = {}

    @ torch.no_grad()
    def momentum_update_k_model_points(self):
        for param_q, param_k in zip(self.model_points.parameters(), self.model_points_k.parameters()):
            param_k.data = param_k.data * self._config['momentum_k'] + param_q.data * (1. - self._config['momentum_k'])

    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def xent_targets(self, A):
        N, M = A.size()
        key = '%s:%s' % (str(A.device), N)

        if key not in self._xent_targets:
            self._xent_targets[key] = torch.arange(A.size(0), device=A.device).long()  # H*W

        return self._xent_targets[key]

    @torch.no_grad()
    def get_matching_matrix(self, pf_1, pf_2, n_points_1, n_points_2):
        start = 0
        ts_start = 0
        matching_matrix = torch.zeros(pf_1.size(0), self._config['cycle_soft_num'], device=pf_1.device).long()
        matching_score = torch.zeros(pf_1.size(0), device=pf_1.device)

        for i, items in enumerate(zip(n_points_1, n_points_2)):
            n_points, ts_n_points = items

            f_1 = pf_1[start:start+n_points]
            f_2 = pf_2[ts_start:ts_start+ts_n_points]

            matching = torch.mm(f_1, f_2.T)

            matching = matching.sort(dim=-1, descending=True)
            matching_matrix[start:start + n_points] = matching[1][:, :self._config['cycle_soft_num']] + ts_start
            matching_score[start:start + n_points] = matching[0][:, :self._config['cycle_soft_num']].mean(dim=-1)

            start += n_points
            ts_start += ts_n_points

        # mask out the features that not matching well
        top_matching_index = torch.sort(matching_score, descending=True)[1]
        partial_top_matching_index = top_matching_index[:int(self._config['cycle_top_percent'] * top_matching_index.size(0))]
        with torch.no_grad():
            mask = torch.zeros(pf_1.size(0), device=pf_1.device)
            mask[partial_top_matching_index] += 1

        return matching_matrix, mask

    def cycle_matching(self, pf_1, pf_2, pf_1_end, batch, one_hot_P):
        start = 0
        ts_start = 0
        jump_matrix = []

        if self._config['temporal_cycle_suppix']:
            B = len(batch['batch_n_points'])

            pairing_points = batch["pairing_points"]
            pf_1_suppix = one_hot_P @ pf_1[pairing_points]
            pf_1_end_suppix = one_hot_P @ pf_1_end[pairing_points]

            assert self._config['normalize_features'], 'We do not support no normalizing now.'
            if self._config['normalize_features']:
                pf_1_suppix = torch.nn.functional.normalize(pf_1_suppix, p=2, dim=1)
                pf_1_end_suppix = torch.nn.functional.normalize(pf_1_end_suppix, p=2, dim=1)

            n, c = pf_1_suppix.size()
            pf_1_suppix = pf_1_suppix.reshape(B, -1, c)
            pf_1_end_suppix = pf_1_end_suppix.reshape(B, -1, c)

        for i, items in enumerate(zip(batch['batch_n_points'], batch['ts_batch_n_points'])):
            n_points, ts_n_points = items
            if not self._config['temporal_cycle_suppix']:
                f_1 = pf_1[start:start+n_points]
                f_1_end = pf_1_end[start:start + n_points]
            else:
                f_1 = pf_1_suppix[i]
                f_1_end = pf_1_end_suppix[i]
                mask = torch.where(f_1[:, 0] != 0)
                f_1 = f_1[mask]
                f_1_end = f_1_end[mask]

            f_2 = pf_2[ts_start:ts_start+ts_n_points]

            # get cycle jump matrix
            if f_1.shape[0] > self._config['cycle_num_matching']:
                rand_idx = np.random.choice(f_1.shape[0], self._config['cycle_num_matching'], replace=False)
                f_1_selected = f_1[rand_idx]
                f_1_end_selected = f_1_end[rand_idx]
            else:
                f_1_selected = f_1
                f_1_end_selected = f_1_end
            A12 = torch.nn.functional.softmax(torch.mm(f_1_selected, f_2.T)/self._config['cycle_t'], dim=-1)
            A21 = torch.nn.functional.softmax(torch.mm(f_2, f_1_end_selected.T)/self._config['cycle_t'], dim=-1)
            jump_matrix.append(torch.mm(A12, A21))

            start += n_points
            ts_start += ts_n_points

        # get cycle loss
        jump_matrix = torch.block_diag(*jump_matrix)
        target = self.xent_targets(jump_matrix)
        logits = torch.log(jump_matrix + 1e-20)
        loss = self.cycle_loss(logits, target).mean()
        return loss

    def triple_cycle(self, pf_1, pf_2, img_f, batch):
        start = 0
        ts_start = 0
        loss = torch.zeros(1).cuda()
        counts = 0

        B, C, H, W = img_f.size()
        B = len(batch['batch_n_points'])
        img_f = img_f.reshape(B, -1, C, H, W)

        for i, items in enumerate(zip(batch['batch_n_points'], batch['ts_batch_n_points'], img_f)):
            n_points, ts_n_points, im_f = items
            pairing_points = batch["pairing_points_org"][i]
            pairing_images = batch["pairing_images_org"][i]
            if len(pairing_images) > self._config['cycle_num_matching']:
                rand_idx = np.random.choice(pairing_points.shape[0], self._config['cycle_num_matching'], replace=False)
                pairing_points = pairing_points[rand_idx]
                pairing_images = pairing_images[rand_idx]

            f_1 = pf_1[start:start + n_points]
            f_2 = pf_2[ts_start:ts_start + ts_n_points]
            im_f = im_f.permute(0, 2, 3, 1)
            im_f = im_f[tuple(pairing_images.T)]  # nc
            f_1 = f_1[pairing_points]

            # get cycle jump matrix
            Aip = torch.nn.functional.softmax(torch.mm(im_f, f_2.T) / self._config['cycle_t'], dim=-1)
            A21 = torch.nn.functional.softmax(torch.mm(f_2, f_1.T) / self._config['cycle_t'], dim=-1)
            jump_matrix = torch.mm(Aip, A21)

            # get cycle loss
            logits = torch.log(jump_matrix + 1e-20)
            # target = pairing_points
            target = self.xent_targets(jump_matrix)
            loss += self.cycle_loss(logits, target).sum()

            start += n_points
            ts_start += ts_n_points
            counts += len(pairing_points)

        return loss / counts

    def triple_cycle_superpixel(self, pf_1, pf_2, img_f, batch, one_hot_P, one_hot_I):
        ts_start = 0
        loss = torch.zeros(1).cuda()
        counts = 0
        B = len(batch['batch_n_points'])

        pairing_points = batch["pairing_points"]
        pf_1_suppix = one_hot_P @ pf_1[pairing_points]
        img_f_suppix = one_hot_I @ img_f.permute(0, 2, 3, 1).flatten(0, 2)

        # norm q and k again
        assert self._config['normalize_features'], 'We do not support no normalizing now.'
        if self._config['normalize_features']:
            pf_1_suppix = torch.nn.functional.normalize(pf_1_suppix, p=2, dim=1)
            img_f_suppix = torch.nn.functional.normalize(img_f_suppix, p=2, dim=1)

        n, c = pf_1_suppix.size()
        pf_1_suppix = pf_1_suppix.reshape(B, -1, c)
        img_f_suppix = img_f_suppix.reshape(B, -1, c)

        for i, items in enumerate(zip(pf_1_suppix, batch['ts_batch_n_points'], img_f_suppix)):
            f_1_suppix, ts_n_points, im_f_suppix = items
            f_2 = pf_2[ts_start:ts_start + ts_n_points]

            mask = torch.where(f_1_suppix[:, 0] != 0)
            f_1_suppix = f_1_suppix[mask]
            im_f_suppix = im_f_suppix[mask]

            # get cycle jump matrix
            Aip = torch.nn.functional.softmax(torch.mm(im_f_suppix, f_2.T) / self._config['cycle_t'], dim=-1)
            A21 = torch.nn.functional.softmax(torch.mm(f_2, f_1_suppix.T) / self._config['cycle_t'], dim=-1)
            jump_matrix = torch.mm(Aip, A21)

            # get cycle loss
            logits = torch.log(jump_matrix + 1e-20)
            # target = pairing_points
            target = self.xent_targets(jump_matrix)
            loss += self.cycle_loss(logits, target).sum()

            ts_start += ts_n_points
            counts += len(im_f_suppix)

        return loss / counts

    def training_step(self, batch, batch_idx):
        if self.current_step == 0 and self._config["resume_path"] is not None:
            self.current_step = self.epoch * self.num_training_batches
        self.current_step += 1

        # forward point stream
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points, output_points_cycle, output_points_triple = self.model_points(sparse_input)
        output_points, output_points_cycle = output_points.F, output_points_cycle.F if output_points_cycle is not None else None
        output_points_triple = output_points_triple.F if output_points_triple is not None else None

        # forward image stream
        self.model_images.eval()
        self.model_images.decoder.train()
        self.model_images.decoder_cycle.train()
        output_images, output_images_triple = self.model_images(batch["input_I"])

        one_hot_P, one_hot_I = self.get_superpixel_one_hot_index(batch, output_images=output_images)

        if self._config['temporal_cycle']:
            ts_sparse_input = ME.SparseTensor(batch["sinput_ts_F"], batch["sinput_ts_C"])
            if self._config['momentum_k'] > 0:
                self.momentum_update_k_model_points()
            if self._config['momentum_k'] or self._config['detach_k']:
                with torch.no_grad():
                    ts_output_points, ts_output_points_cycle, ts_output_points_triple = self.model_points_k(ts_sparse_input)
                    ts_output_points_cycle = ts_output_points_cycle.F
                    ts_output_points = ts_output_points.F
                    ts_output_points_triple = ts_output_points_triple
            else:
                ts_output_points, ts_output_points_cycle, ts_output_points_triple = self.model_points_k(ts_sparse_input)
                ts_output_points_cycle = ts_output_points_cycle.F
                ts_output_points = ts_output_points.F
                ts_output_points_triple = ts_output_points_triple.F
            if self._config['diffStartEnd']:
                sparse_input2 = ME.SparseTensor(batch["sinput_F2"], batch["sinput_C2"])
                output_points2, output_points_cycle2, output_points_triple2 = self.model_points(sparse_input2)
                output_points2, output_points_cycle2 = output_points2.F, output_points_cycle2.F
                output_points_triple2 = output_points_triple2.F
            loss_cycle = self.cycle_matching(output_points_cycle,
                                                                    ts_output_points_cycle,
                                                                    output_points_cycle2 if self._config['diffStartEnd'] else output_points_cycle,
                                                                    batch, one_hot_P)
        else:
            ts_output_points = None
            ts_output_points_cycle = None
            ts_output_points_triple = None
            loss_cycle = 0

        if self._config['reTSFeat']:
            ts_sparse_input = ME.SparseTensor(batch["sinput_ts_F"], batch["sinput_ts_C"])
            ts_output_points, ts_output_points_cycle_recal, ts_output_points_triple = self.model_points(ts_sparse_input)
            ts_output_points = ts_output_points.F
            ts_output_points_cycle_recal = ts_output_points_cycle_recal.F
            ts_output_points_triple = ts_output_points_triple.F

        # permute pf_2 to make it has the same order of pf_1
        if ts_output_points is not None:
            matching_matrix, mask = self.get_matching_matrix(pf_1=output_points_cycle, pf_2=ts_output_points_cycle_recal if self._config['rematching'] else ts_output_points_cycle,
                                                             n_points_1=batch['batch_n_points'], n_points_2=batch['ts_batch_n_points'])
            ts_output_points = ts_output_points[matching_matrix].mean(dim=1)
            ts_output_points = ts_output_points * mask[:, None]


        del batch["sinput_F"]
        del batch["sinput_C"]
        del batch["sinput_ts_F"]
        del batch["sinput_ts_C"]
        del sparse_input
        if self._config['temporal_cycle']:
            del ts_sparse_input

        # each loss is applied independtly on each GPU
        losses = [
            getattr(self, loss)(batch, output_points, output_images, one_hot_P, one_hot_I, ts_output_points)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        if loss_cycle != 0:
            loss = loss + self._config['cycle_weight'] * loss_cycle

        if self._config['triple_cycle']:
            if not self._config['triple_cycle_superpixel']:
                loss_triple_cycle = self.triple_cycle(output_points_triple, ts_output_points_triple, output_images_triple, batch)
            else:
                loss_triple_cycle = self.triple_cycle_superpixel(output_points_triple, ts_output_points_triple, output_images_triple, batch, one_hot_P, one_hot_I)
        else:
            loss_triple_cycle = 0

        if loss_triple_cycle != 0:
            loss = loss + self._config['triple_cycle_weight'] * loss_triple_cycle

        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        if self._config['temporal_cycle']:
            self.log(
                "train_cycle_loss", loss_cycle, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
            if self._config['gradual_cycle']:
                self.log(
                    "cycle_contrast_weight", min(1.0, self.current_step/ (self.trainer.num_training_batches * self._config['num_epochs'] * self._config['gradual_cycle'])), on_step=True, on_epoch=True, prog_bar=True, logger=True,
                    batch_size=self.batch_size
                )
        if self._config['triple_cycle']:
            self.log(
                "train_triple_cycle_loss", loss_triple_cycle, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                batch_size=self.batch_size
            )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def get_superpixel_one_hot_index(self, batch, output_images):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]

        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )
        return one_hot_P, one_hot_I

    def loss(self, batch, output_points, output_images, ts_output_points=None):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)

    def loss_superpixels_average(self, batch, output_points, output_images, one_hot_P, one_hot_I, ts_output_points=None):
        pairing_points = batch["pairing_points"]

        k = one_hot_P @ output_points[pairing_points]
        if ts_output_points is not None:
            one_hot_P_ts = one_hot_P
            ts_k = one_hot_P_ts @ ts_output_points[pairing_points]
            if self._config['gradual_cycle'] <= 0:
                k = k + ts_k
            else:
                k = k + min(1.0, self.current_step/ (self.trainer.num_training_batches * self._config['num_epochs'] * self._config['gradual_cycle'])) * ts_k
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)

        # norm q and k again
        assert self._config['normalize_features'], 'We do not support no normalizing now.'
        if self._config['normalize_features']:
            k = torch.nn.functional.normalize(k, p=2, dim=1)
            q = torch.nn.functional.normalize(q, p=2, dim=1)

        if self._config['contrastive_each_sample']:
            B = len(batch["batch_n_points"])
            n, c = k.size()
            ks = k.reshape(B, -1, c)
            qs = q.reshape(B, -1, c)

            loss = torch.zeros(1, device=k.device)
            for q, k in zip(ks, qs):
                mask = torch.where(k[:, 0] != 0)
                k = k[mask]
                q = q[mask]
                loss += self.criterion(k, q)
            return loss

        # cal the loss in global manner
        if self._config['gather_contrastive_q']:
            assert not self._config['contrastive_each_sample'], 'contrastive_each_sample needs to be false when adopting gather_contrastive_q'
            k = torch.cat(GatherLayer.apply(k), dim=0)
            q = torch.cat(GatherLayer.apply(q), dim=0)
        return self.criterion(k, q)

    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points, _, _ = self.model_points(sparse_input)
        output_points = output_points.F
        self.model_images.eval()
        output_images, _ = self.model_images(batch["input_I"])

        one_hot_P, one_hot_I = self.get_superpixel_one_hot_index(batch, output_images=output_images)

        losses = [
            getattr(self, loss)(batch, output_points, output_images, one_hot_P, one_hot_I)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_images": self.model_images.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )
