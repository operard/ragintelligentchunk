#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 06:03:09 2025

@author: operard
"""

from oci_pdf_lister import OCIPDFObjectStorageLister

# Reemplaza con tus valores reales
compartment_id = ""
bucket_name = ""
namespace_name = ""  # Obténlo de la consola OCI (Object Storage > Buckets > Namespace)

# Crea una instancia de la librería
lister = OCIPDFObjectStorageLister(compartment_id, bucket_name, namespace_name)

# Lista PDFs en el directorio raíz
pdfs_root = lister.list_pdf_files()
print("PDFs en el root:", pdfs_root)

# Lista PDFs en un subdirectorio (e.g., "documentos/")
pdfs_subdir = lister.list_pdf_files(prefix="documentos/")
print("PDFs en documentos/:", pdfs_subdir)


# Lista PDFs
#pdfs = lister.list_pdf_files(prefix="documentos/")
#print("PDFs encontrados:", pdfs)

# Descarga un PDF específico
#if pdfs:
#    lister.download_pdf(pdfs[0], local_file_path="descargado.pdf")  # Descarga el primero a 'descargado.pdf'

# Descarga todos los PDFs en el directorio
lister.download_all_pdfs(prefix="", local_dir="./mis_pdfs")