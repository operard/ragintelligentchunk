#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 06:01:42 2025

oci_pdf_lister.py

@author: operard
"""
import oci
import os  # Para manejar archivos locales

class OCIPDFObjectStorageLister:
    def __init__(self, compartment_id, bucket_name, namespace_name, config_file=None, config_profile=None):
        """
        Inicializa el cliente de Object Storage.
        
        :param compartment_id: ID del compartment donde está el bucket.
        :param bucket_name: Nombre del bucket.
        :param namespace_name: Namespace del tenant (puedes obtenerlo de la consola OCI).
        :param config_file: Ruta al archivo de configuración OCI (opcional, por defecto ~/.oci/config).
        :param config_profile: Perfil en el config file (opcional, por defecto DEFAULT).
        """
        self.compartment_id = compartment_id
        self.bucket_name = bucket_name
        self.namespace_name = namespace_name
        
        # Configura el cliente
        if config_file:
            self.config = oci.config.from_file(config_file, config_profile)
        else:
            self.config = oci.config.from_file()  # Usa ~/.oci/config por defecto
        
        self.object_storage_client = oci.object_storage.ObjectStorageClient(self.config)

    def list_pdf_files(self, prefix=""):
        """
        Lista los archivos PDF en el bucket bajo el prefijo especificado (equivalente a un directorio).
        
        :param prefix: Prefijo del directorio (e.g., "documentos/"). Deja vacío para el root.
        :return: Lista de nombres de archivos PDF.
        """
        try:
            # Lista objetos con el prefijo
            list_objects_response = self.object_storage_client.list_objects(
                namespace_name=self.namespace_name,
                bucket_name=self.bucket_name,
                prefix=prefix,
                fields="name"  # Solo necesitamos el nombre
            )
            
            # Filtra solo archivos que terminen en .pdf
            pdf_files = []
            for obj in list_objects_response.data.objects:
                if obj.name.lower().endswith('.pdf'):
                    pdf_files.append(obj.name)
            
            return pdf_files
        
        except oci.exceptions.ServiceError as e:
            print(f"Error al acceder a Object Storage: {e}")
            return []

    def download_pdf(self, object_name, local_file_path=None):
        """
        Descarga un PDF específico desde el Object Storage.
        
        :param object_name: Nombre completo del objeto (e.g., "documentos/ejemplo.pdf").
        :param local_file_path: Ruta local para guardar el archivo (opcional). Si no se especifica, usa el nombre del objeto.
        :return: True si se descarga exitosamente, False en caso de error.
        """
        try:
            if not local_file_path:
                local_file_path = object_name.split('/')[-1]  # Usa solo el nombre del archivo
            
            # Obtiene el objeto
            get_object_response = self.object_storage_client.get_object(
                namespace_name=self.namespace_name,
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            
            # Guarda el contenido en un archivo local
            with open(local_file_path, 'wb') as f:
                f.write(get_object_response.data.content)
            
            print(f"PDF descargado exitosamente: {local_file_path}")
            return True
        
        except oci.exceptions.ServiceError as e:
            print(f"Error al descargar {object_name}: {e}")
            return False
        except Exception as e:
            print(f"Error general al descargar {object_name}: {e}")
            return False

    def download_all_pdfs(self, prefix="", local_dir="./downloads"):
        """
        Descarga todos los PDFs listados en un prefijo a una carpeta local.
        
        :param prefix: Prefijo del directorio (e.g., "documentos/"). Deja vacío para el root.
        :param local_dir: Directorio local donde guardar los PDFs (se crea si no existe).
        :return: Lista de archivos descargados exitosamente.
        """
        # Crea el directorio local si no existe
        os.makedirs(local_dir, exist_ok=True)
        
        # Lista los PDFs
        pdf_files = self.list_pdf_files(prefix)
        downloaded = []
        
        for pdf in pdf_files:
            local_path = os.path.join(local_dir, pdf.split('/')[-1])  # Usa solo el nombre del archivo
            if self.download_pdf(pdf, local_path):
                downloaded.append(pdf)
        
        print(f"Descargados {len(downloaded)} PDFs de {len(pdf_files)}.")
        return downloaded
