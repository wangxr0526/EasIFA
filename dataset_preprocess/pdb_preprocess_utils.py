from Bio import pairwise2
from Bio.Seq import Seq
import numpy as np
import pandas as pd


def get_active_site_binary(active_site, aa_sequence_len, begain_zero=True):

    active_site_bin = np.zeros((aa_sequence_len,))
    for one_site in active_site:
        if len(one_site) == 1:
            if begain_zero:
                active_site_bin[one_site[0]] = 1
            else:
                active_site_bin[one_site[0] - 1] = 1
        elif len(one_site) == 2:
            b, e = one_site
            if begain_zero:
                # site_indices = [k for k in range(b, e)]
                site_indices = [k for k in range(b, e + 1)]
            else:
                # site_indices = [k-1 for k in range(b, e)]
                site_indices = [k - 1 for k in range(b, e + 1)]
            active_site_bin[site_indices] = 1
        else:
            raise ValueError("The label of active site is not standard !!!")
    return active_site_bin


def get_active_site_types(
    active_site, active_site_type, aa_sequence_len, begain_zero=True
):
    assert len(active_site) == len(active_site_type)
    active_site_types = np.zeros((aa_sequence_len,))
    for one_site, one_type in zip(active_site, active_site_type):
        if len(one_site) == 1:
            if begain_zero:
                active_site_types[one_site[0]] = one_type + 1
            else:
                active_site_types[one_site[0] - 1] = one_type + 1
        elif len(one_site) == 2:
            b, e = one_site
            if begain_zero:
                # site_indices = [k for k in range(b, e)]
                site_indices = [k for k in range(b, e + 1)]
            else:
                # site_indices = [k-1 for k in range(b, e)]
                site_indices = [k - 1 for k in range(b, e + 1)]
            active_site_types[site_indices] = one_type + 1
        else:
            raise ValueError("The label of active site is not standard !!!")
    return active_site_types


def transform_to_active_site_label(active_site_bin):
    active_site_label = []
    active_site_idx = np.argwhere(active_site_bin == 1)
    for idx in active_site_idx:
        b = idx.item()
        e = b + 1
        active_site_label.append([b, e])
    return active_site_label


def map_active_site_for_one(seqA, seqB, active_site_A, begain_zero=True):
    seqA_len = len(seqA)
    active_site_A_bin = get_active_site_binary(active_site_A, seqA_len, begain_zero)
    alignments = pairwise2.align.localxx(seqB, seqA)  # 取最佳结果
    if alignments:
        alignment = alignments[0]
        padded_seqA = alignment[1]
        padded_seqB = alignment[0]
        padded_seqB_arr = np.asanyarray([x for x in padded_seqB])
        padded_active_site_A_bin = np.zeros((len(padded_seqA),))
        org_idx = 0
        for padded_idx, aa in enumerate(padded_seqA):
            if aa != "-":
                padded_active_site_A_bin[padded_idx] = active_site_A_bin[org_idx]
                org_idx += 1
            else:
                pass
        seqB_active_site_bin = padded_active_site_A_bin[padded_seqB_arr != "-"]
    else:
        seqB_active_site_bin = None
    return seqB_active_site_bin


def map_active_site_type_for_one(
    seqA, seqB, active_site_A, active_site_type_A, begain_zero=True
):
    seqA_len = len(seqA)
    active_site_A_type = get_active_site_types(
        active_site_A, active_site_type_A, seqA_len, begain_zero
    )
    alignments = pairwise2.align.localxx(seqB, seqA)  # 取最佳结果
    if alignments:
        alignment = alignments[0]
        padded_seqA = alignment[1]
        padded_seqB = alignment[0]
        padded_seqB_arr = np.asanyarray([x for x in padded_seqB])
        padded_active_site_A_bin = np.zeros((len(padded_seqA),))
        org_idx = 0
        for padded_idx, aa in enumerate(padded_seqA):
            if aa != "-":
                padded_active_site_A_bin[padded_idx] = active_site_A_type[org_idx]
                org_idx += 1
            else:
                pass
        active_site_B_type = padded_active_site_A_bin[padded_seqB_arr != "-"]
    else:
        active_site_B_type = None
    return active_site_B_type


def map_active_site(aa_sequence, split_pdb_aa_sequence_dict, active_site):
    aa_sequence_len = len(aa_sequence)
    active_site_bin = get_active_site_binary(active_site, aa_sequence_len)

    scores = {}
    mapped_active_site_for_split_pdb_aa_sequence = {}

    for chain in split_pdb_aa_sequence_dict:
        aa_sequence_split = split_pdb_aa_sequence_dict[chain]
        alignments = pairwise2.align.localxx(
            aa_sequence_split, aa_sequence
        )  # 取最佳结果
        if alignments:
            alignment = alignments[0]
            scores[chain] = alignment.score
            padded_aa_sequence = alignment[1]
            padded_aa_sequence_split = alignment[0]
            padded_aa_sequence_split_arr = np.asanyarray(
                [x for x in padded_aa_sequence_split]
            )

            padded_active_site_bin = np.zeros((len(padded_aa_sequence),))
            org_idx = 0
            for padded_idx, aa in enumerate(padded_aa_sequence):
                if aa != "-":
                    padded_active_site_bin[padded_idx] = active_site_bin[org_idx]
                    org_idx += 1
                else:
                    pass

            aa_sequence_split_active_site_bin = padded_active_site_bin[
                padded_aa_sequence_split_arr != "-"
            ]

            aa_sequence_split_active_site_label = transform_to_active_site_label(
                aa_sequence_split_active_site_bin
            )
            mapped_active_site_for_split_pdb_aa_sequence[chain] = (
                aa_sequence_split_active_site_label
            )
        else:
            scores[chain] = -1
            mapped_active_site_for_split_pdb_aa_sequence[chain] = []

    results_df = pd.DataFrame()
    results_df["chains"] = list(mapped_active_site_for_split_pdb_aa_sequence.keys())
    results_df["aa_sequence"] = results_df["chains"].apply(
        lambda x: split_pdb_aa_sequence_dict[x]
    )
    results_df["mapped_active_site"] = results_df["chains"].apply(
        lambda x: mapped_active_site_for_split_pdb_aa_sequence[x]
    )
    results_df["scores"] = results_df["chains"].apply(lambda x: scores[x])
    results_df = results_df.sort_values(by="scores", ascending=False).reset_index(
        drop=True
    )
    return results_df


if __name__ == "__main__":

    # aa_sequence = 'MALRWGIVSVGLISSDFTAVLQTLPRSEHQVVAVAARDLSRAKEFAQKHDIPKAYGSYEELAKDPNVEVAYVGTQHPQHKAAVMLCLAAGKAVLCEKPMGVNAAEVREMVTEARSRGLFLMEAIWTRFFPASEALRSVLAQGTLGDLRVARAEFGKNLTHVPRAVDWAQAGGALLDLGIYCVQFISMVFGGQKPEKISVMGRRHETGVDDTVTVLLQYPGEVHGSFTCSITAQLSNTASVSGTKGMAQLLNPCWCPTELVVKGEHKEFLLPPVPKNCNFDNGAGMSYEAKHVRECLRKGLKESPVIPLVESELLADILEEVRRAIGVTFPQDKH'
    # # aa_sequence = 'MSEKYIVTWDMLQIHARKLASRLMPSEQWKGIIAVSRGGLVQPGALLARELGIRHVDTVCISSYDHDNQRELKVLKRAEGDGEGFIVIDDLVDTGGTAVAIREMYPKAHFVTIFAKPAGRPLVDDYVVDIPQDTWIEQPWDMGVVFVPPISGR'

    # split_pdb_aa_sequence_dict = {'X': 'ALRWGIVSVGLISSDFTAVLQTLPRSEHQVVAVAARDLSRAKEFAQKHDIPKAYGSYEELAKDPNVEVAYVGTQHPQHKAAVMLCLAAGKAVLCEKPMGVNAAEVREMVTEARSRGLFLMEAIWTRFFPASEALRSVLAQGTLGDLRVARAEFGKNLTHVPRAVDWAQAGGALLDLGIYCVQFISMVFGGQKPEKISVMGRRHETGVDDTVTVLLQYPGEVHGSFTCSITAQLSNTASVSGTKGMAQLLNPCWCPTELVVKGEHKEFLLPPVPKNCNFDNGAGMSYEAKHVRECLRKGLKESPVIPLVESELLADILEEVRRAIGVTFPQD'}

    # active_site = '[[333, 334]]'  # 这里是从0开始的，和uniprot导出的不一样，uniprot导出位点从1开始
    # active_site = eval(active_site)

    # mapped_results = map_active_site(aa_sequence, split_pdb_aa_sequence_dict,
    #                                  active_site)

    aa_sequence = "MSLAGKEFKLSNGNKIPAVAFGTGTKYFKRGHNDLDKQLIGTLELALRSGFRHIDGAEIYGTNKEIGIALKNVGLNRKDVFITDKYNSGNHTYDGKHSKHQNPYNALKADLEDLGLEYVDLYLIHFPYISEKSHGFDLVEAWRYLERAKNEGLARNIGVSNFTIENLKSILDANTDSIPVVNQIEFSAYLQDQTPGIVEYSQQQGILIEAYGPLGPITQGRPGPLDKVLSKLSEKYKRNEGQILLRWVLQRGILPITTTSKEERINDVLEIFDFELDKEDEDQITKVGKEKTLRQFSKEYSKYD"

    query_aa_sequence = "MRLDGKTALITGSARGIGRAFAEAYVREGARVAIADINLEAARATAAEIGPAACAIALDVTDQASIDRCVAELLDRWGSIDILVNNAALFDLAPIVEITRESYDRLFAINVSGTLFMMQAVARAMIAGGRGGKIINMASQAGRRGEALVGVYCATKAAVISLTQSAGLNLIRHGINVNAIAPGVVDGEHWDGVDAKFADYENLPRGEKKRQVGAAVPFGRMGRAEDLTGMAIFLATPEADYIVAQTYNVDGGNWMS"

    active_site = "[[125], [212, 268], [60], [85]]"
    active_site_type = "[0, 0, 1, 2]"
    active_site = eval(active_site)
    active_site_type = eval(active_site_type)

    mapped_results = map_active_site_type_for_one(
        seqA=aa_sequence,
        seqB=query_aa_sequence,
        active_site_A=active_site,
        active_site_type_A=active_site_type,
        begain_zero=False,
    )
