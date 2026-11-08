"""Hyperstack self-hibernation with a file-based secret and verified VM identity."""
import argparse
import json
import os
from pathlib import Path
import socket
import stat
import urllib.error
import urllib.request

BASE = 'https://infrahub-api.nexgencloud.com/v1/core/virtual-machines'


class Hyperstack:
    def __init__(self):
        if not os.environ.get('HYPERSTACK_VM_ID') or not os.environ.get('HYPERSTACK_API_KEY_FILE'):
            raise ValueError('Set HYPERSTACK_VM_ID and HYPERSTACK_API_KEY_FILE before training')
        self.vm_id = int(os.environ['HYPERSTACK_VM_ID'])
        if self.vm_id <= 0:
            raise ValueError('VM ID must be positive')
        self.key_path = Path(os.environ['HYPERSTACK_API_KEY_FILE']).expanduser()
        mode = self.key_path.stat().st_mode
        if not stat.S_ISREG(mode) or mode & 0o077:
            raise ValueError('API key must be a regular private file (chmod 600)')
        self.key = self.key_path.read_text().strip()
        if not self.key or '\n' in self.key:
            raise ValueError('Invalid API key file')

    def request(self, suffix=''):
        req = urllib.request.Request(f'{BASE}/{self.vm_id}{suffix}',
                                     headers={'api_key': self.key, 'accept': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Hyperstack HTTP {error.code}') from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError('Hyperstack network request failed') from None
        if data.get('status') is False:
            raise RuntimeError('Hyperstack rejected request')
        return data

    def preflight(self):
        data = self.request()
        instance = data.get('instance') or data.get('virtual_machine')
        if instance is None:
            rows = data.get('instances', [])
            instance = next((x for x in rows if int(x['id']) == self.vm_id), None)
        if not instance or int(instance['id']) != self.vm_id:
            raise RuntimeError('API did not return requested VM')
        # Require exact cloud UUID or hostname; never hibernate a merely similar VM.
        cloud_id = Path('/var/lib/cloud/data/instance-id').read_text().strip()
        def values(obj):
            if isinstance(obj, dict):
                return [v for x in obj.values() for v in values(x)]
            if isinstance(obj, list):
                return [v for x in obj for v in values(x)]
            return [str(obj)]
        if cloud_id not in values(instance) and instance.get('name') != socket.gethostname():
            raise RuntimeError('VM identity does not match local cloud UUID/hostname')
        if str(instance.get('status', '')).upper() != 'ACTIVE':
            raise RuntimeError('Current VM is not ACTIVE')
        return {'vm_id': self.vm_id, 'name': instance.get('name'), 'identity_verified': True,
                'cloud_instance_id': cloud_id, 'state': instance['status']}

    def hibernate(self):
        # Retain the existing public IP so the user can reconnect after restore.
        return self.request('/hibernate?retain_ip=true')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(Hyperstack().preflight(), indent=2))
