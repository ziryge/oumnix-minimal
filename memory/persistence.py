"""
"""
import os
import json
import pickle
import gzip
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64
import torch
import numpy as np
try:
    from memory.episodic import EpisodicMemory  # type: ignore
except Exception:  # pragma: no cover
    EpisodicMemory = None  # type: ignore

@dataclass
class LifeFileHeader:
    """
"""
    version: str = "1.0"
    created: float = 0.0
    modified: float = 0.0
    checksum: str = ""
    compressed: bool = True
    encrypted: bool = True
    segments: List[str] = None
    
    def __post_init__(self):
        if self.segments is None:
            self.segments = []
        if self.created == 0.0:
            self.created = time.time()
        self.modified = time.time()

@dataclass
class LifeSegment:
    """
"""
    name: str
    data_type: str  
    checksum: str
    size_bytes: int
    compressed_size: int
    offset: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class CryptoManager:
    """
"""
    def __init__(self, password: Optional[str] = None):
        self.password = password or "default_ai_password_change_me"
        self._fernet = None
        self._setup_crypto()
    
    def _setup_crypto(self):
        """
"""
        password_bytes = self.password.encode()
        # Allow environment-provided salt for deployments; fallback to default
        salt_env = os.environ.get('OUMNIX_STATE_SALT')
        if salt_env is not None:
            try:
                salt = salt_env.encode('utf-8')
            except Exception:
                salt = b'ai_personal_salt_2024'
        else:
            salt = b'ai_personal_salt_2024'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self._fernet = Fernet(key)
    
    def encrypt(self, data: bytes) -> bytes:
        """
"""
        return self._fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
"""
        return self._fernet.decrypt(encrypted_data)
    
    def change_password(self, new_password: str):
        """
"""
        self.password = new_password
        self._setup_crypto()

class DataSerializer:
    """
"""
    
    @staticmethod
    def serialize_tensor(tensor: torch.Tensor) -> bytes:
        """
"""
        buffer = torch.BytesStorage.from_buffer(tensor.detach().cpu().numpy().tobytes())
        metadata = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device)
        }
        
        
        meta_bytes = json.dumps(metadata).encode()
        meta_len = len(meta_bytes).to_bytes(4, 'little')
        
        return meta_len + meta_bytes + tensor.detach().cpu().numpy().tobytes()
    
    @staticmethod
    def deserialize_tensor(data: bytes) -> torch.Tensor:
        """
"""
        
        meta_len = int.from_bytes(data[:4], 'little')
        
        
        meta_bytes = data[4:4+meta_len]
        metadata = json.loads(meta_bytes.decode())
        
        
        tensor_bytes = data[4+meta_len:]
        
        
        dtype_map = {
            'torch.float32': np.float32,
            'torch.float16': np.float16,
            'torch.int64': np.int64,
            'torch.int32': np.int32,
        }
        
        np_dtype = dtype_map.get(metadata['dtype'], np.float32)
        np_array = np.frombuffer(tensor_bytes, dtype=np_dtype)
        np_array = np_array.reshape(metadata['shape'])
        
        return torch.from_numpy(np_array)
    
    @staticmethod
    def serialize_numpy(array: np.ndarray) -> bytes:
        """
"""
        import io
        buffer = io.BytesIO()
        np.save(buffer, array)
        return buffer.getvalue()
    
    @staticmethod
    def deserialize_numpy(data: bytes) -> np.ndarray:
        """
"""
        import io
        buffer = io.BytesIO(data)
        return np.load(buffer)
    
    @staticmethod
    def serialize_pickle(obj: Any) -> bytes:
        """
"""
        return pickle.dumps(obj)
    
    @staticmethod
    def deserialize_pickle(data: bytes) -> Any:
        """
"""
        return pickle.loads(data)
    
    @staticmethod
    def serialize_json(obj: Any) -> bytes:
        """
"""
        return json.dumps(obj, default=str).encode()
    
    @staticmethod
    def deserialize_json(data: bytes) -> Any:
        """
"""
        return json.loads(data.decode())

class LifeFile:
    """
"""
    
    def __init__(self, filepath: str, password: Optional[str] = None):
        self.filepath = Path(filepath)
        self.crypto = CryptoManager(password)
        self.serializer = DataSerializer()
        
        
        self.header = LifeFileHeader()
        self.segments: Dict[str, LifeSegment] = {}
        self.data_cache: Dict[str, Any] = {}
        
        
        self.compression_level = 6
        self.auto_backup = True
        self.max_backups = 5
    
    def add_segment(self, name: str, data: Any, data_type: str = 'auto') -> None:
        """
"""
        
        
        if data_type == 'auto':
            if isinstance(data, torch.Tensor):
                data_type = 'tensor'
            elif isinstance(data, np.ndarray):
                data_type = 'numpy'
            elif isinstance(data, (dict, list, str, int, float, bool)):
                data_type = 'json'
            else:
                data_type = 'pickle'
        
        
        if data_type == 'tensor':
            serialized = self.serializer.serialize_tensor(data)
        elif data_type == 'numpy':
            serialized = self.serializer.serialize_numpy(data)
        elif data_type == 'json':
            serialized = self.serializer.serialize_json(data)
        elif data_type == 'pickle':
            serialized = self.serializer.serialize_pickle(data)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        
        original_size = len(serialized)
        if self.header.compressed:
            compressed = gzip.compress(serialized, compresslevel=self.compression_level)
            compressed_size = len(compressed)
            final_data = compressed
        else:
            compressed_size = original_size
            final_data = serialized
        
        
        checksum = hashlib.sha256(final_data).hexdigest()
        
        
        segment = LifeSegment(
            name=name,
            data_type=data_type,
            checksum=checksum,
            size_bytes=original_size,
            compressed_size=compressed_size,
            offset=0,  
            metadata={
                'compression_ratio': compressed_size / original_size if original_size > 0 else 1.0,
                'timestamp': time.time()
            }
        )
        
        self.segments[name] = segment
        self.data_cache[name] = final_data
        
        
        if name not in self.header.segments:
            self.header.segments.append(name)
        self.header.modified = time.time()
    
    def get_segment(self, name: str) -> Any:
        """
"""
        if name not in self.segments:
            raise KeyError(f"Segment '{name}' not found")
        
        segment = self.segments[name]
        
        
        if name in self.data_cache:
            raw_data = self.data_cache[name]
        else:
            
            raw_data = self._read_segment_from_file(segment)
        
        
        if self.header.compressed:
            decompressed = gzip.decompress(raw_data)
        else:
            decompressed = raw_data
        
        
        expected_checksum = segment.checksum
        actual_checksum = hashlib.sha256(raw_data).hexdigest()
        if expected_checksum != actual_checksum:
            raise ValueError(f"Checksum mismatch for segment '{name}'")
        
        
        if segment.data_type == 'tensor':
            return self.serializer.deserialize_tensor(decompressed)
        elif segment.data_type == 'numpy':
            return self.serializer.deserialize_numpy(decompressed)
        elif segment.data_type == 'json':
            return self.serializer.deserialize_json(decompressed)
        elif segment.data_type == 'pickle':
            return self.serializer.deserialize_pickle(decompressed)
        else:
            raise ValueError(f"Unknown data type: {segment.data_type}")
    
    def _read_segment_from_file(self, segment: LifeSegment) -> bytes:
        """
"""
        with open(self.filepath, 'rb') as f:
            f.seek(segment.offset)
            encrypted_data = f.read(segment.compressed_size)
            return self.crypto.decrypt(encrypted_data)
    
    def save(self) -> None:
        """
"""
        
        if self.auto_backup and self.filepath.exists():
            self._create_backup()
        
        
        temp_path = self.filepath.with_suffix('.life.tmp')
        
        try:
            
            self._write_file(temp_path)
            
            
            if self._verify_file(temp_path):
                
                if self.filepath.exists():
                    self.filepath.unlink()
                temp_path.rename(self.filepath)
            else:
                raise ValueError("File verification failed")
                
        except Exception as e:
            
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _write_file(self, filepath: Path) -> None:
        """
"""
        with open(filepath, 'wb') as f:
            
            current_offset = 0
            
            
            header_space = 4096  
            current_offset += header_space
            
            
            for name in self.header.segments:
                segment = self.segments[name]
                data = self.data_cache[name]
                
                
                encrypted_data = self.crypto.encrypt(data)
                
                
                segment.offset = current_offset
                segment.compressed_size = len(encrypted_data)
                
                
                f.seek(current_offset)
                f.write(encrypted_data)
                
                current_offset += len(encrypted_data)
            
            
            f.flush()
            # compute checksum over data region written after header
            with open(filepath, 'rb') as rf:
                rf.seek(header_space)
                file_data = rf.read()
            self.header.checksum = hashlib.sha256(file_data).hexdigest()
            
            
            header_data = {
                'header': asdict(self.header),
                'segments': {name: asdict(segment) for name, segment in self.segments.items()}
            }
            
            header_json = json.dumps(header_data, indent=2).encode()
            header_compressed = gzip.compress(header_json)
            header_encrypted = self.crypto.encrypt(header_compressed)
            
            
            if len(header_encrypted) > header_space - 8:
                raise ValueError("Header too large")
            
            
            f.seek(0)
            f.write(len(header_encrypted).to_bytes(4, 'little'))
            f.write(b'LIFE')  
            f.write(header_encrypted)
    
    def load(self) -> None:
        """
"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        with open(self.filepath, 'rb') as f:
            
            header_size_bytes = f.read(4)
            header_size = int.from_bytes(header_size_bytes, 'little')
            
            
            magic = f.read(4)
            if magic != b'LIFE':
                raise ValueError("Invalid .life file format")
            
            
            header_encrypted = f.read(header_size)
            header_compressed = self.crypto.decrypt(header_encrypted)
            header_json = gzip.decompress(header_compressed)
            header_data = json.loads(header_json.decode())
            
            
            self.header = LifeFileHeader(**header_data['header'])
            self.segments = {
                name: LifeSegment(**segment_data)
                for name, segment_data in header_data['segments'].items()
            }
            
            
            # read data region starting after header_space (4096)
            f.seek(4096)
            file_data = f.read()
            actual_checksum = hashlib.sha256(file_data).hexdigest()
            
            if actual_checksum != self.header.checksum:
                raise ValueError("File checksum mismatch - file may be corrupted")
        
        
        self.data_cache.clear()
    
    def _verify_file(self, filepath: Path) -> bool:
        """
"""
        try:
            
            temp_life = LifeFile(str(filepath), self.crypto.password)
            temp_life.load()
            # try reading segments if present; tolerate empty
            for name in list(temp_life.header.segments):
                _ = temp_life.segments[name]  # ensure segment metadata present
            return True
        except Exception:
            return False
    
    def _create_backup(self) -> None:
        """
"""
        if not self.filepath.exists():
            return
        
        backup_dir = self.filepath.parent / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        
        timestamp = int(time.time())
        backup_name = f"{self.filepath.stem}_{timestamp}.life.bak"
        backup_path = backup_dir / backup_name
        
        
        import shutil
        shutil.copy2(self.filepath, backup_path)
        
        
        self._cleanup_old_backups(backup_dir)
    
    def _cleanup_old_backups(self, backup_dir: Path) -> None:
        """
"""
        backups = list(backup_dir.glob(f"{self.filepath.stem}_*.life.bak"))
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        
        for backup in backups[self.max_backups:]:
            backup.unlink()
    
    def fsck(self) -> Dict[str, Any]:
        """
"""
        report = {
            'file_exists': self.filepath.exists(),
            'file_size': self.filepath.stat().st_size if self.filepath.exists() else 0,
            'header_valid': False,
            'segments_valid': {},
            'checksum_valid': False,
            'errors': [],
            'warnings': []
        }
        
        if not report['file_exists']:
            report['errors'].append("File does not exist")
            return report
        
        try:
            
            self.load()
            report['header_valid'] = True
            
            
            for name in self.header.segments:
                try:
                    data = self.get_segment(name)
                    report['segments_valid'][name] = True
                except Exception as e:
                    report['segments_valid'][name] = False
                    report['errors'].append(f"Segment '{name}' corrupted: {str(e)}")
            # structural checks: offsets within file
            try:
                size = self.filepath.stat().st_size
                for name, seg in self.segments.items():
                    end_off = seg.offset + seg.compressed_size
                    if seg.offset < 4096:
                        report['warnings'].append(f"Segment '{name}' overlaps header")
                    if end_off > size:
                        report['errors'].append(f"Segment '{name}' exceeds file size (end {end_off} > {size})")
            except Exception:
                pass

            with open(self.filepath, 'rb') as f:
                f.seek(4096)
                file_data = f.read()
                actual_checksum = hashlib.sha256(file_data).hexdigest()
                report['checksum_valid'] = (actual_checksum == self.header.checksum)
                
                if not report['checksum_valid']:
                    report['errors'].append("File checksum mismatch")
            
        except Exception as e:
            report['errors'].append(f"Failed to load file: {str(e)}")
        
        return report
    
    def get_info(self) -> Dict[str, Any]:
        """
"""
        if not self.filepath.exists():
            return {'exists': False}
        
        stat = self.filepath.stat()
        
        info = {
            'exists': True,
            'filepath': str(self.filepath),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': self.header.created,
            'modified': self.header.modified,
            'version': self.header.version,
            'compressed': self.header.compressed,
            'encrypted': self.header.encrypted,
            'segments': []
        }
        
        for name, segment in self.segments.items():
            segment_info = {
                'name': name,
                'type': segment.data_type,
                'size_bytes': segment.size_bytes,
                'compressed_size': segment.compressed_size,
                'compression_ratio': segment.metadata.get('compression_ratio', 1.0),
                'checksum': segment.checksum[:16] + '...'  
            }
            info['segments'].append(segment_info)
        
        return info

class PersistenceManager:
    """
"""
    
    def __init__(self, base_dir: str = ".ai_state", password: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.password = password
        self.life_file = LifeFile(self.base_dir / "ai_state.life", password)
        
        
        self.auto_save_interval = 300  
        self.last_save_time = 0
    
    def save_complete_state(self, 
                           model_state: Dict[str, Any],
                           memory_state: Dict[str, Any],
                           neuro_state: Any,
                           metacognition_state: Dict[str, Any],
                           config: Dict[str, Any],
                           optimizer_state: Optional[Dict[str, Any]] = None,
                           scheduler_state: Optional[Dict[str, Any]] = None,
                           scaler_state: Optional[Dict[str, Any]] = None) -> None:
        """
"""
        # Episodic memory (FAISS) — persist meta + vectors as separate segments (avoid pickling index)
        episodic_obj = None
        try:
            if isinstance(memory_state, dict):
                episodic_obj = memory_state.get('episodic')
        except Exception:
            episodic_obj = None
        if episodic_obj is not None and EpisodicMemory is not None and isinstance(episodic_obj, EpisodicMemory):  # type: ignore[attr-defined]
            self._add_episodic_segments(episodic_obj)
            # Avoid pickling the raw object inside memory_state
            try:
                memory_state = {k: v for k, v in memory_state.items() if k != 'episodic'}
            except Exception:
                pass

        # Core segments
        self.life_file.add_segment('model_weights', model_state, 'pickle')
        self.life_file.add_segment('memory_state', memory_state, 'pickle')
        self.life_file.add_segment('neuro_state', neuro_state, 'pickle')
        self.life_file.add_segment('metacognition_state', metacognition_state, 'pickle')
        self.life_file.add_segment('config', config, 'json')
        if optimizer_state is not None:
            self.life_file.add_segment('optimizer_state', optimizer_state, 'pickle')
        if scheduler_state is not None:
            self.life_file.add_segment('scheduler_state', scheduler_state, 'pickle')
        if scaler_state is not None:
            self.life_file.add_segment('scaler_state', scaler_state, 'pickle')
        
        
        metadata = {
            'save_timestamp': time.time(),
            'save_count': getattr(self, 'save_count', 0) + 1,
            'version': '1.0'
        }
        self.life_file.add_segment('metadata', metadata, 'json')
        
        
        self.life_file.save()
        self.last_save_time = time.time()
        self.save_count = metadata['save_count']
    
    def load_complete_state(self) -> Dict[str, Any]:
        """
"""
        self.life_file.load()
        
        state = {
            'model_weights': self.life_file.get_segment('model_weights'),
            'memory_state': self.life_file.get_segment('memory_state'),
            'neuro_state': self.life_file.get_segment('neuro_state'),
            'metacognition_state': self.life_file.get_segment('metacognition_state'),
            'config': self.life_file.get_segment('config'),
            'metadata': self.life_file.get_segment('metadata')
        }
        # Rehydrate episodic memory if segments are present
        try:
            episodic = self._load_episodic_segments()
            if episodic is not None:
                state['episodic'] = episodic
        except Exception:
            pass
        # optional segments
        try:
            state['optimizer_state'] = self.life_file.get_segment('optimizer_state')
        except Exception:
            pass
        try:
            state['scheduler_state'] = self.life_file.get_segment('scheduler_state')
        except Exception:
            pass
        try:
            state['scaler_state'] = self.life_file.get_segment('scaler_state')
        except Exception:
            pass
        return state

    # ---- Episodic Memory segment helpers ----
    def _add_episodic_segments(self, ep: Any) -> None:
        try:
            # pull simple meta
            meta = {
                'dim': int(getattr(ep, 'dim', 0)),
                'normalize': bool(getattr(ep, 'normalize', False)),
                'metric': str(getattr(ep, 'metric', 'l2')).lower(),
                'store_vectors': bool(getattr(ep, 'store_vectors', True)),
                'texts': list(getattr(ep, 'texts', [])),
            }
            self.life_file.add_segment('episodic_meta', meta, 'json')
            # vectors
            vectors = []
            try:
                vectors = getattr(ep, '_vectors', [])  # list[np.ndarray]
            except Exception:
                vectors = []
            if meta['store_vectors'] and vectors:
                import numpy as _np
                arr = _np.stack(vectors, axis=0).astype('float32')
                self.life_file.add_segment('episodic_vectors', arr, 'numpy')
        except Exception:
            pass

    def _load_episodic_segments(self) -> Optional[Any]:
        try:
            meta = self.life_file.get_segment('episodic_meta')
        except Exception:
            return None
        if meta is None:
            return None
        try:
            dim = int(meta.get('dim', 0))
            normalize = bool(meta.get('normalize', False))
            metric = str(meta.get('metric', 'l2'))
            store_vectors = bool(meta.get('store_vectors', True))
            texts = list(meta.get('texts', []))
            if EpisodicMemory is None:
                return None
            ep = EpisodicMemory(dim=dim, normalize=normalize, store_vectors=store_vectors, metric=metric)  # type: ignore[misc]
            # restore vectors and index
            if store_vectors:
                try:
                    arr = self.life_file.get_segment('episodic_vectors')
                    import numpy as _np
                    if isinstance(arr, _np.ndarray) and arr.size > 0:
                        n = arr.shape[0]
                        t = texts[:n] if len(texts) >= n else texts + [""] * (n - len(texts))
                        ep.add(arr, t)
                except Exception:
                    pass
            # restore texts-only when not storing vectors
            if not store_vectors and texts:
                ep.texts = texts
            return ep
        except Exception:
            return None
    
    def should_auto_save(self) -> bool:
        """
"""
        return (time.time() - self.last_save_time) > self.auto_save_interval
    
    def get_state_info(self) -> Dict[str, Any]:
        """
"""
        return self.life_file.get_info()
    
    def change_password(self, new_password: str) -> None:
        """Rotate encryption password and rewrite LifeFile with backups."""
        # Load existing segments
        try:
            self.life_file.load()
        except Exception:
            # if no file yet, just switch password for future saves
            self.life_file.crypto.change_password(new_password)
            return
        # Read and cache all segments as deserialized objects
        segment_names = list(self.life_file.header.segments)
        deserialized: Dict[str, tuple[Any, str]] = {}
        for name in segment_names:
            seg = self.life_file.segments[name]
            data = self.life_file.get_segment(name)
            deserialized[name] = (data, seg.data_type)
        # Switch password
        self.life_file.crypto.change_password(new_password)
        # Re-add all segments to refresh data_cache with new encryption on save
        self.life_file.segments.clear()
        self.life_file.header.segments = []
        self.life_file.data_cache.clear()
        for name, (data, dtype) in deserialized.items():
            self.life_file.add_segment(name, data, dtype)
        # Save new encrypted file
        self.life_file.save()
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
"""
        return self.life_file.fsck()

__all__ = [
    'PersistenceManager',
    'LifeFile', 
    'CryptoManager',
    'DataSerializer'
]
