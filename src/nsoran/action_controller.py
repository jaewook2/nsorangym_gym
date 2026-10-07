from os import path


class ActionController():
    """
    The ActionController class is responsible for delivering the action to ns-O-RAN.
    In the stand-alone mode, the action is delivered by writing on the appropriate file.
    Multiple channels (TS + ES) can be registered so both CSVs are written before ns-3 reads.
    """
    directory: str
    log_filename: str
    control_filename: str
    channels: dict

    def __init__(self, sim_path, log_filename=None, control_filename=None, header=None,
                 channels=None):
        """Initialize Controller and its files
        Args:
            sim_path (str): the simulation path
            log_filename (str): the name of the file where the action is logged (not read by ns-3)
            control_filename (str): the name of the control file that delivers the action to ns-3
            header (list): fields of the action
            channels (list[dict]): optional extra/override channels, each with
                kind, file, log, header
        """
        self.directory = sim_path
        self.channels = {}
        if channels:
            for channel in channels:
                self._add_channel(channel["kind"], channel["file"], channel["log"],
                                  channel["header"])
        elif control_filename:
            self._add_channel("default", control_filename, log_filename, header)
        else:
            raise ValueError("ActionController needs control_filename or channels")

        self.log_filename = log_filename or next(iter(self.channels.values()))["log"]
        self.control_filename = control_filename or next(iter(self.channels.values()))["file"]

    def _add_channel(self, kind, control_filename, log_filename, header):
        if not control_filename or not log_filename or not header:
            raise ValueError(f"Incomplete control channel '{kind}'")
        self.channels[kind] = {
            "file": control_filename,
            "log": log_filename,
            "header": header,
        }
        with open(path.join(self.directory, log_filename), "w") as file:
            file.write(f"{','.join(header)}\n")
            file.flush()
        open(path.join(self.directory, control_filename), "a").close()

    def create_control_action(self, timestamp: int, actions, kind=None):
        """Applies the control action by writing it in the appropriate file
            timestamp (int) : action's timestamp
            actions [(tuple)]: list of tuples representing the actions to be sent
            kind (str): channel name when multiple files are used (ts, es)
        """
        if not actions:
            return
        if kind is None:
            if len(self.channels) == 1:
                kind = next(iter(self.channels))
            elif "default" in self.channels:
                kind = "default"
            else:
                raise ValueError("kind is required when multiple control channels are configured")
        if kind not in self.channels:
            raise ValueError(f"Unknown control channel '{kind}'. Known: {list(self.channels)}")
        channel = self.channels[kind]
        with open(path.join(self.directory, channel["log"]), "a") as logFile:
            with open(path.join(self.directory, channel["file"]), "w") as file:
                for action in actions:
                    control_action = f"{timestamp},{','.join(map(str, action))}\n"
                    file.write(control_action)
                    logFile.write(control_action)
                file.flush()
                logFile.flush()
