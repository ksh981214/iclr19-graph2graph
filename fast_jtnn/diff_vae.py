import torch
import torch.nn as nn
import torch.nn.functional as F
from mol_tree import Vocab, MolTree
from nnutils import create_var, flatten_tensor, avg_pool
from jtnn_enc import JTNNEncoder
from jtnn_dec import JTNNDecoder
from mpn import MPN
from jtmpn import JTMPN

from chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols
import rdkit
import rdkit.Chem as Chem
import copy, math

class DiffVAE(nn.Module):

    def __init__(self, vocab, args):
        super(DiffVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size = args.hidden_size #hidden state의 차원
        self.rand_size = rand_size = args.rand_size #latent code의 차원

        self.jtmpn = JTMPN(hidden_size, args.depthG) #Junction Tree Message Passing Network , 쉽게 생각해서 임베딩 해주는 애라고 생각
        self.mpn = MPN(hidden_size, args.depthG)     #Message Passing Network
        
        if args.share_embedding: #Encoder와 Decoder의 parameter를 공유할 것이냐? --> 논문에 어디있을까?
            self.embedding = nn.Embedding(vocab.size(), hidden_size)
            self.jtnn = JTNNEncoder(hidden_size, args.depthT, self.embedding)
            self.decoder = JTNNDecoder(vocab, hidden_size, self.embedding, args.use_molatt)
        else:
            self.jtnn = JTNNEncoder(hidden_size, args.depthT, nn.Embedding(vocab.size(), hidden_size))
            self.decoder = JTNNDecoder(vocab, hidden_size, nn.Embedding(vocab.size(), hidden_size), args.use_molatt)

        self.A_assm = nn.Linear(hidden_size, hidden_size, bias=False) #
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(hidden_size, rand_size / 2) # Z를 샘플링 할 때의 Tree와 Graph의 mean, variance를 구할 때 쓰는 linear 
        self.T_var = nn.Linear(hidden_size, rand_size / 2)  # rand size가 latent code의 dim
        self.G_mean = nn.Linear(hidden_size, rand_size / 2)
        self.G_var = nn.Linear(hidden_size, rand_size / 2)
        self.B_t = nn.Sequential(nn.Linear(hidden_size + rand_size / 2, hidden_size), nn.ReLU()) #perturbed vector로 추정
        self.B_g = nn.Sequential(nn.Linear(hidden_size + rand_size / 2, hidden_size), nn.ReLU())
            
    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder) #tree 
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs

    def fuse_noise(self, tree_vecs, mol_vecs):
        tree_eps = create_var( torch.randn(tree_vecs.size(0), 1, self.rand_size / 2) )
        tree_eps = tree_eps.expand(-1, tree_vecs.size(1), -1)
        mol_eps = create_var( torch.randn(mol_vecs.size(0), 1, self.rand_size / 2) )
        mol_eps = mol_eps.expand(-1, mol_vecs.size(1), -1)

        tree_vecs = torch.cat([tree_vecs,tree_eps], dim=-1) 
        mol_vecs = torch.cat([mol_vecs,mol_eps], dim=-1) 
        return self.B_t(tree_vecs), self.B_g(mol_vecs)
    
    #X,Y를 받아서 Z를 샘플링하고 --> perturbed X를 구함 + KL divergence까지
    def fuse_pair(self, x_tree_vecs, x_mol_vecs, y_tree_vecs, y_mol_vecs, jtenc_scope, mpn_scope):
        diff_tree_vecs = y_tree_vecs.sum(dim=1) - x_tree_vecs.sum(dim=1)
        size = create_var(torch.Tensor([le for _,le in jtenc_scope]))
        diff_tree_vecs = diff_tree_vecs / size.unsqueeze(-1)

        diff_mol_vecs = y_mol_vecs.sum(dim=1) - x_mol_vecs.sum(dim=1)
        size = create_var(torch.Tensor([le for _,le in mpn_scope]))
        diff_mol_vecs = diff_mol_vecs / size.unsqueeze(-1)

        diff_tree_vecs, tree_kl = self.rsample(diff_tree_vecs, self.T_mean, self.T_var)
        diff_mol_vecs, mol_kl = self.rsample(diff_mol_vecs, self.G_mean, self.G_var)

        diff_tree_vecs = diff_tree_vecs.unsqueeze(1).expand(-1, x_tree_vecs.size(1), -1)
        diff_mol_vecs = diff_mol_vecs.unsqueeze(1).expand(-1, x_mol_vecs.size(1), -1)
        x_tree_vecs = torch.cat([x_tree_vecs,diff_tree_vecs], dim=-1)
        x_mol_vecs = torch.cat([x_mol_vecs,diff_mol_vecs], dim=-1)

        return self.B_t(x_tree_vecs), self.B_g(x_mol_vecs), tree_kl + mol_kl

    def rsample(self, z_vecs, W_mean, W_var):
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.mean(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def forward(self, x_batch, y_batch, beta):
        x_batch, x_jtenc_holder, x_mpn_holder = x_batch
        y_batch, y_jtenc_holder, y_mpn_holder, y_jtmpn_holder = y_batch

        x_tree_vecs, _, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder) #인코딩의 결과로 tree/mol(graph) vector
        y_tree_vecs, y_tree_mess, y_mol_vecs = self.encode(y_jtenc_holder, y_mpn_holder)

        x_tree_vecs, x_mol_vecs, kl_div = self.fuse_pair(x_tree_vecs, x_mol_vecs, y_tree_vecs, y_mol_vecs, y_jtenc_holder[-1], y_mpn_holder[-1])
        #둘의 dif vector를 구하는 과정, z를 구하고 바로 kl_div를 구함
        
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(y_batch, x_tree_vecs, x_mol_vecs)
        #word_loss
        
        assm_loss, assm_acc = self.assm(y_batch, y_jtmpn_holder, x_mol_vecs, y_tree_mess)

        return word_loss + topo_loss + assm_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, y_tree_mess):
        jtmpn_holder,batch_idx = jtmpn_holder
        fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess)

        x_mol_vecs = x_mol_vecs.sum(dim=1) #average pooling?
        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs) #bilinear
        scores = torch.bmm(
                x_mol_vecs.unsqueeze(1),
                cand_vecs.unsqueeze(-1)
        ).squeeze()
        
        cnt,tot,acc = 0,0,0
        all_loss = []
        for i,mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append( self.assm_loss(cur_score.view(1,-1), label) )
        
        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs):
        #currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root,pred_nodes = self.decoder.decode(x_tree_vecs, x_mol_vecs)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.smiles

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict) #Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vec_pooled = x_mol_vecs.sum(dim=1) #average pooling?
        x_mol_vec_pooled = self.A_assm(x_mol_vec_pooled).squeeze() #bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, x_mol_vec_pooled, pred_nodes, cur_mol, global_amap, [], pred_root, None)
        if cur_mol is None: 
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
        
    def dfs_assemble(self, y_tree_mess, x_mol_vec_pooled, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None

        cand_smiles,cand_amap = zip(*cands)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
        fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])

        scores = torch.mv(cand_vecs, x_mol_vec_pooled)
        _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        #for i in xrange(cand_idx.numel()):
        for i in xrange( min(cand_idx.numel(), 5) ):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            result = True
            for nei_node in children:
                if nei_node.is_leaf: continue
                cur_mol = self.dfs_assemble(y_tree_mess, x_mol_vec_pooled, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node)
                if cur_mol is None: 
                    result = False
                    break
            if result: return cur_mol

        return None
