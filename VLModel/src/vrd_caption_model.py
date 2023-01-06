import torch
from torch import nn
from torch.nn.modules import loss

###
import matplotlib.pyplot as plt
import numpy
def plot_attention(data, i, X_label=None, Y_label=None):
  '''
    Plot the attention model heatmap
    Args:
      data: attn_matrix with shape [ty, tx], cutted before 'PAD'
      X_label: list of size tx, encoder tags
      Y_label: list of size ty, decoder tags
  '''
  fig, ax = plt.subplots(figsize=(92, 92)) # set figure size
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
  
  # Set axis labels
  if X_label != None and Y_label != None:
    X_label = [x_label for x_label in X_label]
    Y_label = [y_label for y_label in Y_label]

    xticks = range(0,len(X_label))
    ax.set_xticks(xticks, minor=False) # major ticks
    ax.set_xticklabels(X_label, minor = False, rotation=45)   # labels should be 'unicode'

    yticks = range(0,len(Y_label))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(Y_label, minor = False)   # labels should be 'unicode'

    ax.grid(True)
  plt.savefig('testblueline_%d.jpg' % i)
  plt.show()  
###

from modeling_t5 import VLT5, PreTrainedModel
class VLT5VRDCaption(VLT5):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.predicate_head = nn.Linear(config.d_model, 9)

        self.sbbox_encode = nn.Linear(4, 64)
        self.obbox_encode = nn.Linear(4, 64)
        self.bbox_cls = nn.Sequential(
                    nn.Linear(64, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.33, inplace=False),
                    nn.Linear(1024, 768)
                )

        self.vrd_cls = self.classifier = nn.Sequential(
                    nn.Linear(config.d_model * 2, 1024),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate, inplace=False),
                    nn.Linear(1024, 9)
                )

        # for p in self.vrd_cls.parameters():
        #     p.requires_grad = False
        # for p in self.sbbox_encode.parameters():
        #     p.requires_grad = False
        # for p in self.obbox_encode.parameters():
        #     p.requires_grad = False
        # for p in self.bbox_cls.parameters():
        #     p.requires_grad = False

        self.init_weights()

    def train_step(self, batch, golden=False, one_step_dec=False):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)
        rel_labels = batch['target_relation_ids'].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            # labels=lm_labels,
            # rel_labels=rel_labels,
            # reduce_loss=reduce_loss
            only_encoder=True
        )

        lm_mask = rel_labels != -100
        B, L = lm_labels.size()

        so_bbox = batch['so_bbox'].to(device)
        so_bbox = so_bbox[lm_mask, :, :]
        s_bbox_h = self.sbbox_encode(so_bbox[:, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)
        

        ## output with golden relation
        if one_step_dec:
            output_with_vrd = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )
        else:
            input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
            output_with_vrd = self(
                input_ids=input_ids_with_vrd,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

        # loss = output['loss']
        loss = output_with_vrd['loss']
        sequence_output = output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        sequence_output = sequence_output[lm_mask, :]

        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        predicate_logits = self.vrd_cls(vrd_h)

        if rel_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            predicate_loss = loss_fct(predicate_logits, rel_labels[lm_mask])
            if not golden:
                loss += predicate_loss        
        
        
        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, beam_with_prompt=False, golden=False, one_step_dec=False, **kwargs):
        # device = next(self.parameters()).device
        # vis_feats = batch['vis_feats'].to(device)
        # input_ids = batch['input_ids'].to(device)
        # vis_pos = batch['boxes'].to(device)

        # output = self.generate(
        #     input_ids=input_ids,
        #     vis_inputs=(vis_feats, vis_pos),
        #     **kwargs
        # )

        # generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        # result = {}
        # result['pred'] = generated_sents

        # return result
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        ### prompt generation
        rel_mask = batch['target_relation_ids'].to(device)
        rel_mask = rel_mask != -100
        rel_mask = rel_mask.int()

        so_bbox = batch['so_bbox'].to(device)
        s_bbox_h = self.sbbox_encode(so_bbox[:, :, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, :, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)


        prompt_output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            only_encoder=True
        )
        sequence_output = prompt_output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        vrd_logits = self.vrd_cls(vrd_h)
        if beam_with_prompt:
            num_beam = kwargs['num_beams']
            vrd_logits = torch.log_softmax(vrd_logits, dim=-1)
            rel_mask_3d = rel_mask.unsqueeze(2).repeat(1,1,9)
            vrd_predict_prob = vrd_logits * rel_mask_3d + (1 - rel_mask_3d)
            vrd_predict_prob = torch.sum(vrd_predict_prob, dim=1)

            vrd_predict_prob = torch.topk(vrd_predict_prob, num_beam, dim=-1)
            vrd_predict = torch.topk(vrd_logits, num_beam, dim=-1)[1]
            t = torch.zeros_like(rel_mask) - 1
            
            seq_prob = []
            seq_l = []
            seq_len = 0
            for i in range(num_beam):
                input_ids_with_promt = vrd_predict[:,:,i] * rel_mask + input_ids * (1 - rel_mask)
                output = self.generate(
                    input_ids=input_ids_with_promt,
                    vis_inputs=(vis_feats, vis_pos),
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
                seq_prob.append(output['sequences_scores'])
                seq_l.append(output['sequences'].transpose(0,1))
                if output['sequences'].shape[1] > seq_len:
                    seq_len = output['sequences'].shape[1]
            seq_prob = torch.stack(seq_prob, dim=1)
            seq_prob = seq_prob + vrd_predict_prob[0]
            best = torch.argmax(seq_prob, dim=-1)
            seqs=torch.nn.utils.rnn.pad_sequence(seq_l, batch_first=True, padding_value=1)
            seqs = seqs.permute(2,0,1)
            seqs = seqs.gather(1, best.unsqueeze(1).unsqueeze(1).repeat(1,1,seqs.shape[2])).squeeze(1)

            generated_sents = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

        else:
            vrd_predict = torch.argmax(vrd_logits, dim=-1)
            ### reverse order !!!
            vrd_predict = - vrd_predict + self.tokenizer.convert_tokens_to_ids('<extra_id_1>')
            input_ids_with_promt = vrd_predict * rel_mask + input_ids * (1 - rel_mask)
            ####
            # replace the
            ###
            # input_ids_with_promt = input_ids
            ####
            
            if golden:
                output = self.generate(
                    input_ids=batch["input_ids_with_vrd"].to(device),
                    vis_inputs=(vis_feats, vis_pos),
                    # return_dict_in_generate=True,
                    # output_attentions=True,
                    **kwargs
                )
            elif one_step_dec:
                output = self.generate(
                    input_ids=input_ids,
                    vis_inputs=(vis_feats, vis_pos),
                    **kwargs
                )
            else:
                output = self.generate(
                    input_ids=input_ids_with_promt,
                    vis_inputs=(vis_feats, vis_pos),
                    **kwargs
                )

            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            # generated_sents = 'test'

        result = {}
        result['pred'] = generated_sents
        result['vrd_class'] = self.tokenizer.convert_ids_to_tokens(vrd_predict[rel_mask != 0])
        result["vrd_target"] = self.tokenizer.convert_ids_to_tokens(batch['input_ids_with_vrd'][batch['target_relation_ids'] != -100])

        return result

    
    def vrd_pretrain_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        rel_labels = batch['target_relation_ids'].to(device)
        # rel_labels = None

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            only_encoder=True
        )

        lm_mask = rel_labels != -100

        so_bbox = batch['so_bbox'].to(device)
        so_bbox = so_bbox[lm_mask, :, :]
        s_bbox_h = self.sbbox_encode(so_bbox[:, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)
        

        ### output with golden relation
        # input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
        # output_with_vrd = self(
        #     input_ids=input_ids_with_vrd,
        #     vis_inputs=(vis_feats, vis_pos),
        #     reduce_loss=reduce_loss
        # )

        # loss = output['loss']
        # loss = output_with_vrd['loss']
        sequence_output = output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        sequence_output = sequence_output[lm_mask, :]

        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        predicate_logits = self.vrd_cls(vrd_h)
        # sequence_output = self.dropout(sequence_output)
        # predicate_logits = self.predicate_head(sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        predicate_loss = loss_fct(predicate_logits, rel_labels[lm_mask])
        loss = predicate_loss
        # loss = loss / 2
        
        
        
        result = {
            'loss': loss
        }
        return result

class VLT5VRDCaptionEx(VLT5):
    def __init__(self, config, vrd_encoder, cap_model):
        super().__init__(config)
        self.cap_model = cap_model
        self.vrd_encoder = vrd_encoder
        self.dropout = nn.Dropout(config.dropout_rate)
        self.predicate_head = nn.Linear(config.d_model, 9)

        self.sbbox_encode = nn.Linear(4, 64)
        self.obbox_encode = nn.Linear(4, 64)
        self.bbox_cls = nn.Sequential(
                    nn.Linear(64, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.33, inplace=False),
                    nn.Linear(1024, 768)
                )

        self.vrd_cls = self.classifier = nn.Sequential(
                    nn.Linear(config.d_model * 2, 1024),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate, inplace=False),
                    nn.Linear(1024, 9)
                )

        # self.step = 2
        # if self.step == 1:
        #     for p  in self.cap_model.parameters():
        #         p.requires_grad = False
        # else:
        #     for p  in self.vrd_encoder.parameters():
        #         p.requires_grad = False
        #     for p in self.vrd_cls.parameters():
        #         p.requires_grad = False
        #     for p in self.sbbox_encode.parameters():
        #         p.requires_grad = False
        #     for p in self.obbox_encode.parameters():
        #         p.requires_grad = False
        #     for p in self.bbox_cls.parameters():
        #         p.requires_grad = False

        # self.init_weights()

    def train_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)
        rel_labels = batch['target_relation_ids'].to(device)

        reduce_loss = True
        output = self.vrd_encoder(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            # labels=lm_labels,
            # rel_labels=rel_labels,
            # reduce_loss=reduce_loss
            # only_encoder=True
        )

        lm_mask = rel_labels != -100
        B, L = lm_labels.size()

        so_bbox = batch['so_bbox'].to(device)
        so_bbox = so_bbox[lm_mask, :, :]
        s_bbox_h = self.sbbox_encode(so_bbox[:, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)
        

        ## output with golden relation
        input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
        output_with_vrd = self.cap_model(
            input_ids=input_ids_with_vrd,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        # loss = output['loss']
        loss_mask = output_with_vrd['loss']
        sequence_output = output[0][:, :input_ids.size(1), :]
        sequence_output = sequence_output[lm_mask, :]

        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        predicate_logits = self.vrd_cls(vrd_h)

        if rel_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            predicate_loss = loss_fct(predicate_logits, rel_labels[lm_mask])
            # if self.step == 1:
            #     loss = predicate_loss
            # else:
            #     loss = loss_mask
            loss = loss_mask + predicate_loss
        
        
        
        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, beam_with_prompt=False, **kwargs):
        # device = next(self.parameters()).device
        # vis_feats = batch['vis_feats'].to(device)
        # input_ids = batch['input_ids'].to(device)
        # vis_pos = batch['boxes'].to(device)

        # output = self.generate(
        #     input_ids=input_ids,
        #     vis_inputs=(vis_feats, vis_pos),
        #     **kwargs
        # )

        # generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        # result = {}
        # result['pred'] = generated_sents

        # return result
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        ### prompt generation
        rel_mask = batch['target_relation_ids'].to(device)
        rel_mask = rel_mask != -100
        rel_mask = rel_mask.int()

        so_bbox = batch['so_bbox'].to(device)
        s_bbox_h = self.sbbox_encode(so_bbox[:, :, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, :, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)


        prompt_output = self.vrd_encoder(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            # only_encoder=True
        )
        sequence_output = prompt_output[0][:, :input_ids.size(1), :]
        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        vrd_logits = self.vrd_cls(vrd_h)
        if beam_with_prompt:
            num_beam = kwargs['num_beams']
            vrd_logits = torch.log_softmax(vrd_logits, dim=-1)
            rel_mask_3d = rel_mask.unsqueeze(2).repeat(1,1,9)
            vrd_predict_prob = vrd_logits * rel_mask_3d + (1 - rel_mask_3d)
            vrd_predict_prob = torch.sum(vrd_predict_prob, dim=1)

            vrd_predict_prob = torch.topk(vrd_predict_prob, num_beam, dim=-1)
            vrd_predict = torch.topk(vrd_logits, num_beam, dim=-1)[1]
            t = torch.zeros_like(rel_mask) - 1
            
            seq_prob = []
            seq_l = []
            seq_len = 0
            for i in range(num_beam):
                input_ids_with_promt = vrd_predict[:,:,i] * rel_mask + input_ids * (1 - rel_mask)
                output = self.generate(
                    input_ids=input_ids_with_promt,
                    vis_inputs=(vis_feats, vis_pos),
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
                seq_prob.append(output['sequences_scores'])
                seq_l.append(output['sequences'].transpose(0,1))
                if output['sequences'].shape[1] > seq_len:
                    seq_len = output['sequences'].shape[1]
            seq_prob = torch.stack(seq_prob, dim=1)
            seq_prob = seq_prob + vrd_predict_prob[0]
            best = torch.argmax(seq_prob, dim=-1)
            seqs=torch.nn.utils.rnn.pad_sequence(seq_l, batch_first=True, padding_value=1)
            seqs = seqs.permute(2,0,1)
            seqs = seqs.gather(1, best.unsqueeze(1).unsqueeze(1).repeat(1,1,seqs.shape[2])).squeeze(1)

            generated_sents = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

        else:
            vrd_predict = torch.argmax(vrd_logits, dim=-1)
            ### reverse order !!!
            vrd_predict = - vrd_predict + self.cap_model.tokenizer.convert_tokens_to_ids('<extra_id_1>')
            input_ids_with_promt = vrd_predict * rel_mask + input_ids * (1 - rel_mask)
            # input_ids_with_promt = input_ids
            ####
            
            output = self.cap_model.generate(
                input_ids=input_ids_with_promt,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )

            generated_sents = self.cap_model.tokenizer.batch_decode(output, skip_special_tokens=True)
            # generated_sents = 'test'

        result = {}
        result['pred'] = generated_sents
        result['vrd_class'] = self.cap_model.tokenizer.convert_ids_to_tokens(vrd_predict[rel_mask != 0])
        result["vrd_target"] = self.cap_model.tokenizer.convert_ids_to_tokens(batch['input_ids_with_vrd'][batch['target_relation_ids'] != -100])

        # result['vrd_class'] = vrd_predict[rel_mask != 0]

        return result

    
    def vrd_pretrain_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        rel_labels = batch['target_relation_ids'].to(device)
        # rel_labels = None

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            only_encoder=True
        )

        lm_mask = rel_labels != -100

        so_bbox = batch['so_bbox'].to(device)
        so_bbox = so_bbox[lm_mask, :, :]
        s_bbox_h = self.sbbox_encode(so_bbox[:, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)
        

        ### output with golden relation
        # input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
        # output_with_vrd = self(
        #     input_ids=input_ids_with_vrd,
        #     vis_inputs=(vis_feats, vis_pos),
        #     reduce_loss=reduce_loss
        # )

        # loss = output['loss']
        # loss = output_with_vrd['loss']
        sequence_output = output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        sequence_output = sequence_output[lm_mask, :]

        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        predicate_logits = self.vrd_cls(vrd_h)
        # sequence_output = self.dropout(sequence_output)
        # predicate_logits = self.predicate_head(sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        predicate_loss = loss_fct(predicate_logits, rel_labels[lm_mask])
        loss = predicate_loss
        # loss = loss / 2
        
        
        
        result = {
            'loss': loss
        }
        return result


from modeling_bart import VLBart, PreTrainedModel
class VLBartVRDCaption(VLBart):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.predicate_head = nn.Linear(config.d_model, 9)

        self.sbbox_encode = nn.Linear(4, 64)
        self.obbox_encode = nn.Linear(4, 64)
        self.bbox_cls = nn.Sequential(
                    nn.Linear(64, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.33, inplace=False),
                    nn.Linear(1024, 768)
                )

        self.vrd_cls = self.classifier = nn.Sequential(
                    nn.Linear(config.d_model * 2, 1024),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate, inplace=False),
                    nn.Linear(1024, 9)
                )
        self.alpha = nn.Parameter(torch.Tensor(1))
        with torch.no_grad():
            self.alpha[0] = 0.5
        # for p in self.vrd_cls.parameters():
        #     p.requires_grad = False
        # for p in self.sbbox_encode.parameters():
        #     p.requires_grad = False
        # for p in self.obbox_encode.parameters():
        #     p.requires_grad = False
        # for p in self.bbox_cls.parameters():
        #     p.requires_grad = False

        self.init_weights()

    def train_step(self, batch, golden=False, one_step_dec=False):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)
        rel_labels = batch['target_relation_ids'].to(device)
        # rel_labels = None

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        lm_mask = rel_labels != -100
        B, L = lm_labels.size()

        so_bbox = batch['so_bbox'].to(device)
        so_bbox = so_bbox[lm_mask, :, :]
        s_bbox_h = self.sbbox_encode(so_bbox[:, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)
        

        ### output with golden relation
        if one_step_dec:
            output_with_vrd = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )
        else:
            input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
            output_with_vrd = self(
                input_ids=input_ids_with_vrd,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

        # loss = output['loss']
        loss_mask = output_with_vrd['loss']
        sequence_output = output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        sequence_output = sequence_output[lm_mask, :]

        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        predicate_logits = self.vrd_cls(vrd_h)
        # sequence_output = self.dropout(sequence_output)
        # predicate_logits = self.predicate_head(sequence_output)

        if rel_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            predicate_loss = loss_fct(predicate_logits, rel_labels[lm_mask])
            if not golden:
                loss = self.alpha * predicate_loss + (1 - self.alpha) + loss_mask
                # loss = predicate_loss + loss_mask
            # loss = loss / 2
        
        
        
        result = {
            'loss': loss
        }
        return result

    def vrd_pretrain_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        rel_labels = batch['target_relation_ids'].to(device)
        # rel_labels = None

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            reduce_loss=reduce_loss
        )

        lm_mask = rel_labels != -100

        so_bbox = batch['so_bbox'].to(device)
        so_bbox = so_bbox[lm_mask, :, :]
        s_bbox_h = self.sbbox_encode(so_bbox[:, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)
        

        ### output with golden relation
        # input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
        # output_with_vrd = self(
        #     input_ids=input_ids_with_vrd,
        #     vis_inputs=(vis_feats, vis_pos),
        #     reduce_loss=reduce_loss
        # )

        # loss = output['loss']
        # loss = output_with_vrd['loss']
        sequence_output = output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        sequence_output = sequence_output[lm_mask, :]

        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        predicate_logits = self.vrd_cls(vrd_h)
        # sequence_output = self.dropout(sequence_output)
        # predicate_logits = self.predicate_head(sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        predicate_loss = loss_fct(predicate_logits, rel_labels[lm_mask])
        loss = predicate_loss
        # loss = loss / 2
        
        
        
        result = {
            'loss': loss
        }
        return result


    def test_step(self, batch, beam_with_prompt=False, golden=False, one_step_dec=False, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
        vis_pos = batch['boxes'].to(device)

        ### prompt generation
        rel_mask = batch['target_relation_ids'].to(device)
        rel_mask = rel_mask != -100
        rel_mask = rel_mask.int()

        so_bbox = batch['so_bbox'].to(device)
        s_bbox_h = self.sbbox_encode(so_bbox[:, :, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, :, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)


        prompt_output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos)
        )
        sequence_output = prompt_output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        vrd_logits = self.vrd_cls(vrd_h)
        if beam_with_prompt:
            num_beam = kwargs['num_beams']
            vrd_logits = torch.log_softmax(vrd_logits, dim=-1)
            rel_mask_3d = rel_mask.unsqueeze(2).repeat(1,1,9)
            vrd_predict_prob = vrd_logits * rel_mask_3d + (1 - rel_mask_3d)
            vrd_predict_prob = torch.sum(vrd_predict_prob, dim=1)

            vrd_predict_prob = torch.topk(vrd_predict_prob, num_beam, dim=-1)
            vrd_predict = torch.topk(vrd_logits, num_beam, dim=-1)[1]
            t = torch.zeros_like(rel_mask) - 1
            
            seq_prob = []
            seq_l = []
            seq_len = 0
            for i in range(num_beam):
                input_ids_with_promt = vrd_predict[:,:,i] * rel_mask + input_ids * (1 - rel_mask)
                output = self.generate(
                    input_ids=input_ids_with_promt,
                    vis_inputs=(vis_feats, vis_pos),
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
                seq_prob.append(output['sequences_scores'])
                seq_l.append(output['sequences'].transpose(0,1))
                if output['sequences'].shape[1] > seq_len:
                    seq_len = output['sequences'].shape[1]
            seq_prob = torch.stack(seq_prob, dim=1)
            seq_prob = seq_prob + vrd_predict_prob[0]
            best = torch.argmax(seq_prob, dim=-1)
            seqs=torch.nn.utils.rnn.pad_sequence(seq_l, batch_first=True, padding_value=1)
            seqs = seqs.permute(2,0,1)
            seqs = seqs.gather(1, best.unsqueeze(1).unsqueeze(1).repeat(1,1,seqs.shape[2])).squeeze(1)

            generated_sents = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

        else:
            vrd_predict = torch.argmax(vrd_logits, dim=-1)
            ### reverse order !!!
            vrd_predict = - vrd_predict + self.tokenizer.convert_tokens_to_ids('<extra_id_1>')
            _input_ids_with_promt = vrd_predict * rel_mask + input_ids * (1 - rel_mask)
            ####
            # replace the
            # input_ids_with_promt = torch.cat((input_ids_with_vrd[:,:9],_input_ids_with_promt[:,4:]),dim=1)
            input_ids_with_promt = _input_ids_with_promt
            ###
            # input_ids_with_promt = input_ids
            ####
            
            if golden:
                output = self.generate(
                    input_ids=batch["input_ids_with_vrd"].to(device),
                    vis_inputs=(vis_feats, vis_pos),
                    return_dict_in_generate=True,
                    # output_attentions=True,
                    **kwargs
                )
            elif one_step_dec:
                output = self.generate(
                    input_ids=input_ids,
                    vis_inputs=(vis_feats, vis_pos),
                    **kwargs
                )
            else:
                output = self.generate(
                    input_ids=input_ids_with_promt,
                    vis_inputs=(vis_feats, vis_pos),
                    return_dict_in_generate=True,
                    # output_attentions=True,
                    **kwargs
                )

            ###
            # for i in range(5):
            #     attentions = torch.sum(output["encoder_attentions"][i][0][:,1:,1:],0) / 12
            #     res = attentions.to("cpu").detach().numpy()
            #     plot_attention(res, i)
            ###

            # generated_sents = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
            generated_sents = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
            # generated_sents = 'test'

        result = {}
        result['pred'] = generated_sents
        result['vrd_class'] = self.tokenizer.convert_ids_to_tokens(vrd_predict[rel_mask != 0])
        # result["vrd_target"] = self.tokenizer.convert_ids_to_tokens(batch['input_ids_with_vrd'][:, 9:][batch['target_relation_ids'][:,4:] != -100])
        result["vrd_target"] = self.tokenizer.convert_ids_to_tokens(batch['input_ids_with_vrd'][batch['target_relation_ids'] != -100])
        return result


    def test_only(self, batch, beam_with_prompt=False, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        # input_ids = batch['input_ids_with_vrd'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        ### prompt generation
        # rel_mask = batch['target_relation_ids'].to(device)
        # rel_mask = rel_mask != -100
        # rel_mask = rel_mask.int()

        # so_bbox = batch['so_bbox'].to(device)
        # s_bbox_h = self.sbbox_encode(so_bbox[:, :, 0, :])
        # o_bbox_h = self.obbox_encode(so_bbox[:, :, 1, :])
        # bbox_h = s_bbox_h + o_bbox_h
        # bbox_h = self.bbox_cls(bbox_h)


        # prompt_output = self(
        #     input_ids=input_ids,
        #     vis_inputs=(vis_feats, vis_pos)
        # )
        # sequence_output = prompt_output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        # vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        # vrd_logits = self.vrd_cls(vrd_h)
        # vrd_predict = torch.argmax(vrd_logits, dim=-1)
        ### reverse order !!!
        # vrd_predict = - vrd_predict + self.tokenizer.convert_tokens_to_ids('<extra_id_1>')
        # input_ids_with_promt = vrd_predict * rel_mask + input_ids * (1 - rel_mask)
        # input_ids_with_promt = input_ids
        ####
        
        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            return_dict_in_generate=True,
            output_attentions=True,
            **kwargs
        )

        ###
        # for i in range(5):
        #     attentions = torch.sum(output["encoder_attentions"][i][0][:,1:,1:],0) / 12
        #     res = attentions.to("cpu").detach().numpy()
        #     plot_attention(res, i)
        ###

        generated_sents = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
        # generated_sents = 'test'

        result = {}
        result['pred'] = generated_sents
        # result['vrd_class'] = vrd_predict[rel_mask != 0]

        return result

class VLBartVRDCaptionEx(PreTrainedModel):
    def __init__(self, config, vrd_model, cap_model):
        super().__init__(config)
        self.cap_model = cap_model
        self.vrd_encoder = vrd_model
        self.dropout = nn.Dropout(config.dropout_rate)
        self.predicate_head = nn.Linear(config.d_model, 9)

        self.sbbox_encode = nn.Linear(4, 64)
        self.obbox_encode = nn.Linear(4, 64)
        self.bbox_cls = nn.Sequential(
                    nn.Linear(64, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.33, inplace=False),
                    nn.Linear(1024, 768)
                )

        self.vrd_cls = self.classifier = nn.Sequential(
                    nn.Linear(config.d_model * 2, 1024),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate, inplace=False),
                    nn.Linear(1024, 9)
                )

        self.step = 2
        if self.step == 1:
            for p  in self.cap_model.parameters():
                p.requires_grad = False
        else:
            for p  in self.vrd_encoder.parameters():
                p.requires_grad = False
            for p in self.vrd_cls.parameters():
                p.requires_grad = False
            for p in self.sbbox_encode.parameters():
                p.requires_grad = False
            for p in self.obbox_encode.parameters():
                p.requires_grad = False
            for p in self.bbox_cls.parameters():
                p.requires_grad = False

        # self.init_weights()

    def train_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)
        rel_labels = batch['target_relation_ids'].to(device)
        # rel_labels = None

        reduce_loss = True
        output = self.vrd_encoder(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        lm_mask = rel_labels != -100
        B, L = lm_labels.size()

        so_bbox = batch['so_bbox'].to(device)
        so_bbox = so_bbox[lm_mask, :, :]
        s_bbox_h = self.sbbox_encode(so_bbox[:, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)
        

        ### output with golden relation
        input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
        output_with_vrd = self.cap_model(
            input_ids=input_ids_with_vrd,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        # loss = output['loss']
        loss_mask = output_with_vrd['loss']
        sequence_output = output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        sequence_output = sequence_output[lm_mask, :]

        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        predicate_logits = self.vrd_cls(vrd_h)
        # sequence_output = self.dropout(sequence_output)
        # predicate_logits = self.predicate_head(sequence_output)

        if rel_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            predicate_loss = loss_fct(predicate_logits, rel_labels[lm_mask])
            if self.step == 1:
                loss = predicate_loss
            else:
                loss = loss_mask
            # loss = loss / 2
        
        
        
        result = {
            'loss': loss
        }
        return result

    def vrd_pretrain_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        rel_labels = batch['target_relation_ids'].to(device)
        # rel_labels = None

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            reduce_loss=reduce_loss
        )

        lm_mask = rel_labels != -100

        so_bbox = batch['so_bbox'].to(device)
        so_bbox = so_bbox[lm_mask, :, :]
        s_bbox_h = self.sbbox_encode(so_bbox[:, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)
        

        ### output with golden relation
        # input_ids_with_vrd = batch['input_ids_with_vrd'].to(device)
        # output_with_vrd = self(
        #     input_ids=input_ids_with_vrd,
        #     vis_inputs=(vis_feats, vis_pos),
        #     reduce_loss=reduce_loss
        # )

        # loss = output['loss']
        # loss = output_with_vrd['loss']
        sequence_output = output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        sequence_output = sequence_output[lm_mask, :]

        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        predicate_logits = self.vrd_cls(vrd_h)
        # sequence_output = self.dropout(sequence_output)
        # predicate_logits = self.predicate_head(sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        predicate_loss = loss_fct(predicate_logits, rel_labels[lm_mask])
        loss = predicate_loss
        # loss = loss / 2
        
        
        
        result = {
            'loss': loss
        }
        return result


    def test_step(self, batch, beam_with_prompt=False, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        ### prompt generation
        rel_mask = batch['target_relation_ids'].to(device)
        rel_mask = rel_mask != -100
        rel_mask = rel_mask.int()

        so_bbox = batch['so_bbox'].to(device)
        s_bbox_h = self.sbbox_encode(so_bbox[:, :, 0, :])
        o_bbox_h = self.obbox_encode(so_bbox[:, :, 1, :])
        bbox_h = s_bbox_h + o_bbox_h
        bbox_h = self.bbox_cls(bbox_h)


        prompt_output = self.vrd_encoder(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos)
        )
        sequence_output = prompt_output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        vrd_logits = self.vrd_cls(vrd_h)
        if beam_with_prompt:
            num_beam = kwargs['num_beams']
            vrd_logits = torch.log_softmax(vrd_logits, dim=-1)
            rel_mask_3d = rel_mask.unsqueeze(2).repeat(1,1,9)
            vrd_predict_prob = vrd_logits * rel_mask_3d + (1 - rel_mask_3d)
            vrd_predict_prob = torch.sum(vrd_predict_prob, dim=1)

            vrd_predict_prob = torch.topk(vrd_predict_prob, num_beam, dim=-1)
            vrd_predict = torch.topk(vrd_logits, num_beam, dim=-1)[1]
            t = torch.zeros_like(rel_mask) - 1
            
            seq_prob = []
            seq_l = []
            seq_len = 0
            for i in range(num_beam):
                input_ids_with_promt = vrd_predict[:,:,i] * rel_mask + input_ids * (1 - rel_mask)
                output = self.generate(
                    input_ids=input_ids_with_promt,
                    vis_inputs=(vis_feats, vis_pos),
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
                seq_prob.append(output['sequences_scores'])
                seq_l.append(output['sequences'].transpose(0,1))
                if output['sequences'].shape[1] > seq_len:
                    seq_len = output['sequences'].shape[1]
            seq_prob = torch.stack(seq_prob, dim=1)
            seq_prob = seq_prob + vrd_predict_prob[0]
            best = torch.argmax(seq_prob, dim=-1)
            seqs=torch.nn.utils.rnn.pad_sequence(seq_l, batch_first=True, padding_value=1)
            seqs = seqs.permute(2,0,1)
            seqs = seqs.gather(1, best.unsqueeze(1).unsqueeze(1).repeat(1,1,seqs.shape[2])).squeeze(1)

            generated_sents = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

        else:
            vrd_predict = torch.argmax(vrd_logits, dim=-1)
            ### reverse order !!!
            vrd_predict = - vrd_predict + self.cap_model.tokenizer.convert_tokens_to_ids('<extra_id_1>')
            input_ids_with_promt = vrd_predict * rel_mask + input_ids * (1 - rel_mask)
            # input_ids_with_promt = input_ids
            ####
            
            output = self.cap_model.generate(
                input_ids=input_ids_with_promt,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )

            generated_sents = self.cap_model.tokenizer.batch_decode(output, skip_special_tokens=True)
            # generated_sents = 'test'

        result = {}
        result['pred'] = generated_sents
        result['vrd_class'] = self.cap_model.tokenizer.convert_ids_to_tokens(vrd_predict[rel_mask != 0])
        result["vrd_target"] = self.cap_model.tokenizer.convert_ids_to_tokens(batch['input_ids_with_vrd'][batch['target_relation_ids'] != -100])

        return result

    def test_only(self, batch, beam_with_prompt=False, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        # input_ids = batch['input_ids_with_vrd'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        ### prompt generation
        # rel_mask = batch['target_relation_ids'].to(device)
        # rel_mask = rel_mask != -100
        # rel_mask = rel_mask.int()

        # so_bbox = batch['so_bbox'].to(device)
        # s_bbox_h = self.sbbox_encode(so_bbox[:, :, 0, :])
        # o_bbox_h = self.obbox_encode(so_bbox[:, :, 1, :])
        # bbox_h = s_bbox_h + o_bbox_h
        # bbox_h = self.bbox_cls(bbox_h)


        # prompt_output = self(
        #     input_ids=input_ids,
        #     vis_inputs=(vis_feats, vis_pos)
        # )
        # sequence_output = prompt_output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
        # vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
        # vrd_logits = self.vrd_cls(vrd_h)
        # vrd_predict = torch.argmax(vrd_logits, dim=-1)
        ### reverse order !!!
        # vrd_predict = - vrd_predict + self.tokenizer.convert_tokens_to_ids('<extra_id_1>')
        # input_ids_with_promt = vrd_predict * rel_mask + input_ids * (1 - rel_mask)
        # input_ids_with_promt = input_ids
        ####
        
        output = self.cap_model.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            return_dict_in_generate=True,
            output_attentions=True,
            **kwargs
        )

        ###
        # for i in range(5):
        #     attentions = torch.sum(output["encoder_attentions"][i][0][:,1:,1:],0) / 12
        #     res = attentions.to("cpu").detach().numpy()
        #     plot_attention(res, i)
        ###

        generated_sents = self.cap_model.tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
        # generated_sents = 'test'

        result = {}
        result['pred'] = generated_sents
        # result['vrd_class'] = vrd_predict[rel_mask != 0]

        return result