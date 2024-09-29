# satellite.py

import datetime as dt
from typing import Optional


class Satellite:
    """
    A class to represent a satellite.

    Attributes
    ----------
    name : str
        The name of the satellite.
    norad_id : Optional[int]
        The NORAD ID of the satellite.
    epoch_utc_dt : Optional[datetime.datetime]
        The epoch UTC time of the satellite.
    """

    def __init__(
        self,
        name: str,
        norad_id: Optional[int] = None,
        epoch_utc_dt: Optional[dt.datetime] = None,
    ) -> None:
        """
        Initializes a Satellite instance.

        Parameters
        ----------
        name : str
            The name of the satellite.
        norad_id : Optional[int], optional
            The NORAD ID of the satellite, by default None.
        epoch_utc_dt : Optional[datetime.datetime], optional
            The epoch UTC datetime of the satellite, by default None.
        """
        self.name = name
        self.norad_id = norad_id
        self.epoch_utc_dt = epoch_utc_dt

    def __str__(self) -> str:
        """
        Returns a string representation of the Satellite instance.

        Returns
        -------
        str
            A human-readable string describing the satellite.
        """
        return f"Satellite {self.name}, NORAD ID: {self.norad_id or 'Unknown'}"

    def __repr__(self) -> str:
        """
        Returns an unambiguous string representation of the Satellite instance.

        Returns
        -------
        str
            A detailed string for debugging that includes the class name and attributes.
        """
        return (
            f"Satellite(name={self.name!r}, "
            f"norad_id={self.norad_id!r}, "
            f"epoch_utc_dt={self.epoch_utc_dt!r})"
        )

    def get_epoch_iso_string(self, sep: Optional[str] = "T") -> str:
        """
        Return the `Satellite` class attribute `epoch_utc_dt` as a string, formatted according to ISO.

        The full format looks like 'YYYY-MM-DD HH:MM:SS.mmmmmm'.
        By default, the fractional part is omitted if self.microsecond == 0.

        If self.tzinfo is not None, the UTC offset is also attached, giving
        a full format of 'YYYY-MM-DD HH:MM:SS.mmmmmm+HH:MM'.

        Parameters
        ----------
        sep : Optional[str], optional
            Optional argument sep specifies the separator between date and time, default 'T'.

        Returns
        -------
        str
            Return the `Satellite` class attribute `epoch_utc_dt` time formatted according to ISO.
        """
        return self.epoch_utc_dt.isoformat(sep=sep)
