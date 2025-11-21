from pathlib import Path

from DIRAC.Resources.Storage.FileStorage import FileStorage  # type: ignore[import-untyped]
from DIRACCommon.Core.Utilities.ReturnValues import S_OK  # type: ignore[import-untyped]


class DataManagerMock:
    """Interface for managing data operations via DIRAC Storage and File Catalogs.

    The ``DataManager`` provides a unified interface to handle file transfers,
    replica management, and catalog registration. It delegates data operations
    to both DIRAC Storage Elements and File Catalog implementations.

    Parameters
    ----------
    file_catalog : str, optional
        Name of the file catalog to use. If set to ``"LocalFileCatalog"``,
        initializes a local catalog and storage element for testing or local use.

    Notes
    -----
    This class acts as a thin abstraction layer over DIRAC's Data Management
    components.
    """

    def __init__(self):
        self.base_storage_path = "filecatalog"
        self.storage_element = FileStorage("local", {"Path": self.base_storage_path})

    def getFile(self, lfn, destinationDir=".", sourceSE=None):
        """Download a file (or files) from a DIRAC Storage Element.

        Parameters
        ----------
        lfn : str | list[str]
            LFN(s) to retrieve.
        destinationDir : str, optional
            Directory where the file(s) should be downloaded.
        sourceSE : Any, optional
            Specific Storage Element to use for retrieval. Defaults to the
            instance’s configured storage element.

        Returns
        -------
        list[Path]
            List of downloaded file names.

        Raises
        ------
        ValueError
            If the provided LFN argument is not a string or list.
        FileNotFoundError
            If any file cannot be retrieved from the storage element.
        """
        if isinstance(lfn, list):
            lfns = lfn
        elif isinstance(lfn, str):
            lfns = [lfn]
        else:
            raise ValueError()

        if not sourceSE:
            sourceSE = self.storage_element
        for lfn in lfns:
            res = sourceSE.getFile(str(lfn).removeprefix("lfn:"), destinationDir)
            if not res["OK"]:
                raise FileNotFoundError(
                    f"Could not download lfn {lfn} : {res['Message']}"
                )
        return S_OK([Path(lfn).name for lfn in lfns])

    def putAndRegister(
        self,
        lfn,
        fileName,
        diracSE=None,
        guid=None,
        path=None,
        checksum=None,
        overwrite=None,
    ):
        """Upload and register a file in the storage element and catalog.

        Parameters
        ----------
        lfn : str
            LFN under which to register the file.
        fileName : str | Path
            Path to the file to upload.
        diracSE : Any, optional
            Specific Storage Element to use. Defaults to the instance’s configured one. Ignored in local testing
        guid : str, optional
            Globally unique identifier for the file. Currently unused.
        path : str, optional
            Target path under the storage element. Defaults to the LFN.
        checksum : str, optional
            Optional checksum for validation. Currently unused.
        overwrite : bool, optional
            Whether to overwrite existing entries. Currently unused.

        Notes
        -----
        This method currently delegates to :meth:`put` and does not update
        the file catalog.
        """

        # TODO use file catalog as well
        return self.put(lfn, fileName, diracSE, path)

    def put(self, lfn, fileName, diracSE=None, path=None):
        """Upload a file to the storage element.

        Parameters
        ----------
        lfn : str
            LFN to associate with the file.
        fileName : str | Path
            Path to the file to be uploaded.
        diracSE : Any, optional
            Specific storage element to use. Defaults to the configured one. Ignored in local testing
        path : str, optional
            Target directory path within the storage element.

        Raises
        ------
        RuntimeError
            If no storage element is configured.
        """
        se = self.storage_element
        if not se:
            raise RuntimeError("No Storage Element defined")
        if not path:
            path = str(lfn).removeprefix("lfn:")
        dest = str(Path(path) / Path(fileName).name)
        se.putFile({dest: fileName})
        return S_OK({"Successful": [lfn], "Failed": []})
