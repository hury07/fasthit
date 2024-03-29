"""Defines abstract base explorer class."""
from typing import Dict, Optional, Tuple
import os
import json
import abc

import time
import tqdm
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import fasthit


class Explorer(abc.ABC):
    """
    Abstract base explorer class.

    Run explorer through the `run` method. Implement subclasses
    by overriding `propose_sequences` (do not override `run`).
    """
    def __init__(
        self,
        name: str,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int,
        model_queries_per_round: int,
        starting_sequence: str,
        log_file: Optional[str] = None,
        seed: Optional[int] = 42,
    ):
        """
        Create an Explorer.

        Args:
            model: Model of ground truth that the explorer will use to help guide
                sequence proposal.
            name: A human-readable name for the explorer (may include parameter values).
            rounds: Number of rounds to run for (a round consists of sequence proposal,
                ground truth fitness measurement of proposed sequences, and retraining
                the model).
            expmt_queries_per_round: Number of sequences to propose for measurement from
                ground truth per round.
            model_queries_per_round: Number of allowed "in-silico" model evaluations
                per round.
            starting_sequence: Sequence from which to start exploration.
            log_file: .csv filepath to write output.

        """
        assert expmt_queries_per_round <= model_queries_per_round
        ###
        self._name = name
        self._encoder = encoder
        self._model = model

        self._rounds = rounds
        self._expmt_queries_per_round = expmt_queries_per_round
        self._model_queries_per_round = model_queries_per_round
        self._starting_sequence = starting_sequence

        self._log_file = log_file
        if self._log_file is not None:
            dir_path, _ = os.path.split(self._log_file)
            os.makedirs(dir_path, exist_ok=True)
        
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Propose a list of sequences to be measured in the next round.

        This method will be overriden to contain the explorer logic for each explorer.

        Args:
            measured_sequences: A pandas dataframe of all sequences that have been
            measured by the ground truth so far. Has columns "sequence",
            "true_score", "model_score", and "round".

        Returns:
            A tuple containing the proposed sequences and their scores
                (according to the model).

        """
        pass

    @abc.abstractmethod
    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        pass

    def _log(
        self,
        measured_data: pd.DataFrame,
        metadata: Dict,
        current_round: int,
        verbose: bool,
        round_start_time: float,
    ) -> None:
        if self._log_file is not None:
            with open(self._log_file, "w") as f:
                # First write metadata
                json.dump(metadata, f)
                f.write("\n")

                # Then write pandas dataframe
                measured_data.to_csv(f, index=False)

        if verbose:
            print(
                f"round: {current_round}, top: {measured_data['true_score'].max()}, "
                f"time: {time.time() - round_start_time:02f}s"
            )

    def run(
        self,
        landscape: fasthit.Landscape,
        init_seqs: np.ndarray = None,
        init_seqs_file: str = None,
        verbose: bool = True,
        eval_models: bool = False,
        testset: np.ndarray = None,
        eval_file: str = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the exporer.

        Args:
            landscape: Ground truth fitness landscape.
            verbose: Whether to print output or not.

        """
        round = 0
        self._model.cost = 0

        # Metadata about run that will be used for logging purposes
        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self._name,
            "encoder_name": self._encoder.name,
            "model_name": self._model.name,
            "landscape_name": landscape.name,
            "rounds": self._rounds,
            "expmt_queries_per_round": self._expmt_queries_per_round,
            "model_queries_per_round": self._model_queries_per_round,
        }

        # Initial sequences and their scores
        measured_data = pd.DataFrame(
            {
                "sequence": self._starting_sequence,
                "model_score": np.nan,
                "true_score": landscape.get_fitness([self._starting_sequence]),
                "round": round,
                "model_cost": 0,
                "measurement_cost": 1,
            }
        )
        self._log(measured_data, metadata, round, verbose, time.time())
        
        if init_seqs is not None or init_seqs_file is not None:
            if init_seqs_file is not None:
                assert init_seqs_file.endswith(".csv")
                df = pd.read_csv(init_seqs_file)
                init_seqs = df["Variants"].to_numpy()
            ###
            true_score = landscape.get_fitness(init_seqs)
            df = pd.DataFrame(
                {
                    "sequence": init_seqs,
                    "true_score": true_score,
                }
            )
            df.dropna(inplace=True)
            true_score = df["true_score"].to_numpy()
            seqs = df["sequence"].values
            ###
            round = 1
            measured_data = measured_data.append(
                pd.DataFrame(
                    {
                        "sequence": seqs,
                        "model_score": np.nan,
                        "true_score": true_score,
                        "round": round,
                        "model_cost": 0,
                        "measurement_cost": len(seqs),
                    }
                )
            )
            self._log(measured_data, metadata, round, verbose, time.time())
        training_data = measured_data
        
        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        range_iterator = range if verbose else tqdm.trange
        eval_data = pd.DataFrame()
        for r in range_iterator(round + 1, self._rounds + 1):
            round_start_time = time.time()

            # print("Encoding...")
            encodings = self._encoder.encode(training_data["sequence"].to_list())
            # print("Training...")
            labels = training_data["true_score"].to_numpy()
            self._model.train(encodings, labels)

            ###
            if eval_models:
                encodings = self._encoder.encode(testset[:, 0])
                evals = self._model.get_fitness(encodings)
                eval_data = eval_data.append(
                    pd.DataFrame(
                        {
                            "sequence": testset[:,0],
                            "model_score": evals,
                            "true_score": testset[:,1],
                            "round": r,
                        }
                    )
                )
                with open(eval_file, "w") as f:
                    # First write metadata
                    json.dump(metadata, f)
                    f.write("\n")
                    # Then write pandas dataframe
                    eval_data.to_csv(f, index=False)
            ###

            if verbose:
                print(f'Round {r} proposing...')
            measured_data, seqs, preds = self.propose_sequences(measured_data)
            if len(seqs) > self._expmt_queries_per_round:
                warnings.warn(
                    "Must propose <= `self._expmt_queries_per_round` sequences per round"
                )
            ###
            if verbose:
                print(f'Round {r} measuring...')
            true_score = landscape.get_fitness(seqs)
            ###
            df = pd.DataFrame(
                {
                    "sequence": seqs,
                    "true_score": true_score,
                    "preds": preds,
                }
            )
            df.dropna(inplace=True)
            true_score = df["true_score"].to_numpy()
            preds = df["preds"].to_numpy()
            seqs = df["sequence"].values
            ###
            measured_data = measured_data.append(
                pd.DataFrame(
                    {
                        "sequence": seqs,
                        "model_score": preds,
                        "true_score": true_score,
                        "round": r,
                        "model_cost": self._model.cost,
                        "measurement_cost": len(measured_data) + len(seqs),
                    }
                )
            )
            self._log(measured_data, metadata, r, verbose, round_start_time)

            training_data = self.get_training_data(measured_data)

        return measured_data, metadata

    @property
    def encoder(self):
        return self._encoder
    
    @property
    def model(self):
        return self._model

    @property
    def starting_sequence(self):
        return self._starting_sequence
        
    @property
    def expmt_queries_per_round(self):
        return self._expmt_queries_per_round
    
    @property
    def model_queries_per_round(self):
        return self._model_queries_per_round
