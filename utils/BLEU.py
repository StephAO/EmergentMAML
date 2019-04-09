import string
import numpy as np

class BLEU:
    
    @staticmethod
    def get_score(candidates, references, N=2, weights=None, 
        apply_smoothing=True, type="sentence"):
        
        assert(len(candidates) == len(references))
        assert(weights == None or len(weights) == N)
        
        r = 0.0
        c = 0.0
        
        n_count_clip = np.zeros(N)
        n_count_unclip = np.zeros(N)
        
        bleu_score = 0.0
        
        for c in range(len(candidates)):
            curr_candidate = candidates[c]
            curr_references = references[c]
            
            curr_candidate = BLEU._tokenize(curr_candidate)
            curr_references = [BLEU._tokenize(ref) for ref in curr_references]
            
            references_len = [len(ref) for ref in curr_references]
            best_match_idx = np.argmin(np.abs(np.array(references_len) - len(curr_candidate)))
            best_match_len = references_len[best_match_idx]
            
            r += best_match_len
            c += len(curr_candidate)
            
            sent_brevity_penalty = min(0.0, 1 - (best_match_len / len(curr_candidate)))
            
            sent_bleu_score = 0.0
            for n in range(N):
                
                clipped_counts = 0.0
                unclip_counts = 0.0
                
                cand_counts = {}
                BLEU._get_counts(curr_candidate, cand_counts, n + 1)
                
                ref_counts = {}
                for r in range(len(curr_references)):
                    curr_ref_counts = {}
                    BLEU._get_counts(curr_references[r], curr_ref_counts, n + 1)
                    # merge counts
                    for (k, v) in curr_ref_counts.items():
                        ref_counts[k] = max(v, ref_counts.get(k, 0))
                
                for (k, v) in cand_counts.items():
                    clipped_counts += min(ref_counts.get(k, 0), v)
                    unclip_counts += v 
                    
                n_count_clip[n] += clipped_counts
                n_count_unclip[n] += unclip_counts
                
                if apply_smoothing or (clipped_counts != 0 and unclip_counts != 0):
                    clipped_counts = max(1e-9, clipped_counts)
                    unclip_counts = max(1e-9, unclip_counts)
                    
                    p_n = np.log(clipped_counts / unclip_counts)
                    if weights is None:
                        p_n = (1/N) * p_n
                    else:
                        p_n = weights[N-1] * p_n
                
                sent_bleu_score += p_n
                
            bleu_score += np.exp(sent_brevity_penalty + sent_bleu_score)
        
        if type == "sentence":
            bleu_score = bleu_score / len(candidates)
            return bleu_score
        elif type == "corpus":
            
            brevity_penalty = min(0.0, 1.0 - (r/c))
            if apply_smoothing:
                n_count_clip[n_count_clip == 0] = 1e-9
                n_count_unclip[n_count_unclip == 0] = 1e-9
            else:
                idx = np.logical_and(n_count_clip != 0, n_count_unclip != 0)
                n_count_clip = n_count_clip[idx]
                n_count_unclip = n_count_unclip[idx]
            
            percision = np.log(n_count_clip / n_count_unclip)
            if weights is None:
                percision = (1/N) * percision
            else:
                percision = np.array(weights) * percision
                
            percision = np.sum(percision)
            bleu_score = np.exp(brevity_penalty + percision)
            
            return bleu_score
            
        else:
            return 0.0
    
    @staticmethod
    def _tokenize(sent):
        """
        Return the tokens of the given sentence sent
        """
        if isinstance(sent, list):
            return sent
        tokens = sent.translate(str.maketrans('', '', string.punctuation))
        return tokens.lower().split()
        
        
    @staticmethod
    def _get_counts(tokens, counts, N):
        i = 0
        while (i + N) <= len(tokens):
            curr_ngram = tuple(tokens[i:(i + N)])
            counts[curr_ngram] = counts.get(curr_ngram, 0) + 1
            i += 1

if __name__ == "__main__":
    
    # code for testing
    hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 
        'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 
        'the', 'party']
    hyp1 = " ".join(hyp1)
    
    ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 
    'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
    ref1a = " ".join(ref1a)
    
    ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    'guarantees', 'the', 'military', 'forces', 'always',  'being', 'under', 
    'the', 'command', 'of', 'the', 'Party']
    ref1b = " ".join(ref1b)
        
    ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the','army', 
    'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']
    ref1c = " ".join(ref1c)
    
    hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was', 'interested', 
    'in', 'world', 'history']
    hyp2 = " ".join(hyp2)
    
    ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',  
    'because', 'he', 'read', 'the', 'book']
    ref2a = " ".join(ref2a)
    
    list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    hypotheses = [hyp1, hyp2]
    
    b_score_corpus = BLEU.get_score(hypotheses,list_of_references, N=4, type="corpus")
    b_score_sent = BLEU.get_score(hypotheses,list_of_references, N=4, type="sentence")
    
    print("BLEU score corpus ", b_score_corpus)
    print("BLEU score sentence ", b_score_sent)
        
        