###
import math
import os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import numpy as np
import pandas as pd
import torch
from typing import List, Optional

_filedir = os.path.dirname(os.path.abspath(__file__))

class ESM(object):
    encodings = pd.DataFrame(
        {
            "encoder": ["esm-1b", "esm-msa-1"],
            "model": ["esm1b_t33_650M_UR50S", "esm_msa1_t12_100M_UR50S"],
            "n_features": [1280, 768],
        }
    )
    encodings.set_index("encoder", inplace=True)

    def __init__(
        self,
        protein_name: str,
        alphabet: str,
        encoding: str,
        wt_seq: str,
        target_python_inds: int,
        output: str = os.getcwd(),
    ) -> None:
        super().__init__()
        self.protein_name = protein_name
        self.alphabet = alphabet
        
        assert encoding in ["esm-1b", "esm-msa-1"]
        self.encoding = self.encodings.loc[encoding]
        
        self.wt_seq = wt_seq
        self.target_python_inds = target_python_inds
        self.n_positions_combined = len(target_python_inds)
        self.output = output
    
    def encode(
        self,
        sequences: List[str],
        batch_size: Optional[int] = None
    ) -> np.array:
        ### [bsz, seq_length, n_features]
        """
        Encodes a given combinatorial space using tape.
        Unlike Georgiev and one-hot encodings, these encodings are context-
        aware. To save on system RAM, this task is split into n_batches number
        of batches.
        """
        # Build fasta files
        n_batches = math.ceil(len(sequences) / batch_size)
        fasta_filenames = self._build_fastas(sequences, n_batches)
        # Create a list to store the names of the raw embedding files
        extracted_embeddings = [None for _ in range(n_batches)]
        # Get a temporary filename for storing batches
        temp_filename = os.path.join(self.output, "TempOutputs_esm.pt")
        # Loop over the number of batches
        for i, fasta_filename in enumerate(fasta_filenames):
            # Run TAPE to get the transformer embeddings
            _ = subprocess.run(
                [
                    "python", "/".join([_filedir, "esm/extract.py"]),
                    self.encoding["model"], fasta_filename, self.output,
                    "--include", "per_tok"
                ]
            )
            # Load the raw embedding file that was generated
            raw_embeddings = torch.load(temp_filename)
            # Extract just the indices we care about
            extracted_embeddings[i] = [
                raw_embedding["representations"][33][self.target_python_inds, :].numpy()
                for raw_embedding in raw_embeddings
            ]
            # Delete the temporary outputs file
            os.remove(fasta_filename)
            os.remove(temp_filename)
        
        # Return the extracted embeddings, concatenating along the first dimension
        return np.concatenate(extracted_embeddings)
    
    def _build_fastas(self, sequences, n_batches):
        """
        The TAPE program requiers a fasta file as an input. It will then encode
        all proteins in the input fasta file. This function takes the input
        fasta file and builds a new fasta file containing all possible variants
        in the combinatorial space. To save on system RAM, this task is split into
        n_batches number of batches.
        """
        # Convert the wt sequence to a list
        wt_list = list(self.wt_seq)
        # Create a list to store all fasta filenames in
        fasta_filenames = [None for _ in range(n_batches)]
        # Loop over the number of batches
        for i, combo_batch in enumerate(np.array_split(sequences, n_batches)):
            # Create a filename for the file we will use to store fasta data
            fasta_filename = os.path.join(self.output,
                                          "{}_Batch{}_Variants_esm.fasta".format(self.protein_name, i))
            # Record the fasta_filename
            fasta_filenames[i] = fasta_filename
            # Create a list in which we will store SeqRecords
            temp_seqs = [None for _ in range(len(combo_batch))]
            # Build fasta for the batch
            for j, combo in enumerate(combo_batch):
                # Make a copy of the wild type list
                temp_seq = wt_list.copy()
                # Create a list to store the variant name
                var_name = [None for _ in range(self.n_positions_combined)]
                # Loop over the target python indices and set new amino acids
                for k, (aa, ind) in enumerate(zip(combo, self.target_python_inds)):
                    # Replace the WT amino acids with the new ones
                    temp_seq[ind] = aa
                    # Format the variant name (OldaaIndNewaa)
                    var_name[k] = "{}{}{}".format(self.wt_seq[k], ind + 1, aa)
                # Create and store a SeqRecords object
                variant_name = f"{self.protein_name}_{'-'.join(var_name)}"
                temp_seqs[j] = SeqRecord(
                    Seq("".join(temp_seq)), id = variant_name,  description=""
                )
            # Write fasta sequences of the variant combinations to the file
            with open(fasta_filename, "w") as f:
                SeqIO.write(temp_seqs, f, "fasta")

        # Return the filename of the fasta sequences
        return fasta_filenames