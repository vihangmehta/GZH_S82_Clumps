from useful import *
from utils_DR14 import *

def getSpecMetadata(specobjid):

    query = doObjectSearch(specObjId=specobjid)
    query = parseSkyServerQuickLookOutput(query)

    try:
        specobjid = specobjid
        primary   = query["SpectralData"]["primary"]
        otherspec = query["SpectralData"]["otherspec"]
    except KeyError:
        print(specobjid)
        specobjid = specobjid
        primary   = -99
        otherspec = -99
    return (specobjid,primary,otherspec)

def customAdd(table):

    custom_specObjIds = [1689976647551313920,453778925533816832]
    _table = np.recarray(len(custom_specObjIds),dtype=table.dtype.descr)

    for i,specObjId in enumerate(custom_specObjIds):

        query = doObjectSearch(specObjId=specObjId)
        query = parseSkyServerQuickLookOutput(query)

        _table[i]["plate"]         = query["SpectralData"]["plate"]
        _table[i]["mjd"]           = query["SpectralData"]["mjd"]
        _table[i]["fiberid"]       = query["SpectralData"]["fiberid"]
        _table[i]["ra"]            = query["MetaData"]["ra"]
        _table[i]["dec"]           = query["MetaData"]["dec"]
        _table[i]["run2d"]         = query["SpectralData"]["run2d"]
        _table[i]["specobjid"]     = query["MetaData"]["specObjId"]
        _table[i]["redshift"]      = query["SpectralData"]["redshift_z"]
        _table[i]["redshift_err"]  = query["SpectralData"]["redshift_err"]
        _table[i]["redshift_flag"] = -99.
        _table[i]["class"]         = query["SpectralData"]["objclass"]
        _table[i]["subclass"]      = ""

    return rfn.stack_arrays((table,_table),usemask=False,asrecarray=True,autoconvert=False)

def mergeSpecSearch():

    df = pd.read_csv("DR14/specSearch/gzh_sdss_clumpy_specSearch.csv")

    table = np.recarray(len(df),dtype=[("plate",int),("mjd",int),("fiberid",int),
                                       ("run2d","<U10"),("specobjid","<U20"),
                                       ("ra",float),("dec",float),
                                       ("redshift",float),("redshift_err",float),("redshift_flag",int),
                                       ("class","<U15"),("subclass","<U25"),
                                       ("primary",int),("otherspec",int)])

    table["plate"]         = df["plate"]
    table["mjd"]           = df["mjd"]
    table["fiberid"]       = df["fiberid"]
    table["ra"]            = df["ra"]
    table["dec"]           = df["dec"]
    table["run2d"]         = [_[1:-1] for _ in df["run2d"]]
    table["specobjid"]     = [_[9:-2] for _ in df["specobj_id"]]
    table["redshift"]      = df["z"]
    table["redshift_err"]  = df["zerr"]
    table["redshift_flag"] = df["zwarning"]
    table["class"]         = [_[1:-1] for _ in df["class"]]
    table["subclass"]      = [_[1:-1] for _ in df["subclass"]]

    iunique = np.unique(table["specobjid"],return_index=True)[1]
    table = table[iunique]

    metadata = Parallel(n_jobs=10,verbose=10,backend="multiprocessing")(delayed(getSpecMetadata)(_) for _ in table["specobjid"])
    metadata = np.array(metadata,dtype=[("specobjid","<U20"),("primary",int),("otherspec",int)])
    assert (table["specobjid"] == metadata["specobjid"]).all()
    table["primary"]   = metadata["primary"]
    table["otherspec"] = metadata["otherspec"]

    table = customAdd(table)
    fitsio.writeto("DR14/specSearch/gzh_sdss_clumpy_specSearch.fits",table,overwrite=True)

if __name__ == '__main__':

    mergeSpecSearch()
