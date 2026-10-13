# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 17:26:43 2025

@author: ALBERTO
"""

from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.colors import Colormap

import torch
from  torchvision.utils import make_grid
from torchvision.transforms import functional


def colorize_mnist(img):
    """Assigns a random RGB color tint to a grayscale MNIST digit.
    
    Takes a 1-channel (H, W) or (1, H, W) tensor and returns a (3, H, W) tensor
    where each pixel is tinted by a random color. The background stays black
    and the digit strokes are colored.
    
    Args:
        img: A single MNIST image tensor of shape (1, H, W) with values in [0, 1].
    
    Returns:
        A (3, H, W) tensor with the colored digit.
    """
    c = torch.rand(3, 1, 1) * 0.6 + 0.4  # Random bright color
    return img.repeat(3, 1, 1) * c

def _enhance_contrast(images: torch.Tensor, gamma: float = 1.0, 
                      bg_threshold: float = 0.0) -> torch.Tensor:
    """Mejora el contraste de las imágenes generadas mediante normalización
    min-max por imagen y corrección gamma opcional.
    
    La normalización min-max estira los valores de cada imagen al rango
    completo [0, 1], lo que mejora drásticamente la visibilidad de los
    dígitos coloreados cuyo rango dinámico original puede ser muy estrecho.
    
    La normalización se aplica de forma CONJUNTA sobre todos los canales
    para preservar las relaciones de color (hue) originales. Si se normalizara
    cada canal independientemente, los colores se distorsionarían.
    
    La corrección gamma (gamma < 1) aclara los tonos medios para hacer
    más visibles los trazos tenues sin saturar las zonas brillantes.
    
    bg_threshold permite suprimir el fondo: los píxeles cuya intensidad
    (media de canales) sea inferior al umbral se ponen a 0, produciendo
    un fondo negro limpio que hace resaltar los dígitos.
    
    Args:
        images: Tensor de forma (N, C, H, W) con valores en [0, 1].
        gamma: Exponente de corrección gamma. 
               gamma < 1 aclara la imagen, gamma > 1 la oscurece.
               gamma = 1.0 no aplica corrección.
        bg_threshold: Umbral de intensidad para suprimir el fondo.
                      0.0 = sin supresión, 0.05-0.15 = limpieza suave.
    
    Returns:
        Tensor con el contraste mejorado, valores en [0, 1].
    """
    enhanced = []
    for img in images:
        # Normalización min-max CONJUNTA (todos los canales a la vez)
        # para preservar las relaciones de color
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-6:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = torch.zeros_like(img)
        
        # Supresión de fondo: poner a negro los píxeles de baja intensidad
        if bg_threshold > 0.0:
            intensity = img.mean(dim=0, keepdim=True)  # (1, H, W)
            mask = (intensity > bg_threshold).float()
            img = img * mask
        
        # Corrección gamma opcional
        if gamma != 1.0:
            img = torch.pow(img.clamp(min=1e-8), gamma)
        
        enhanced.append(img)
    
    return torch.stack(enhanced)



def plot_image_grid( 
    images: torch.Tensor, 
    figsize: tuple,
    n_rows: int,
    n_cols: int,
    padding: int = 2,
    pad_value: float = 0.0,
    cmap: Colormap = "gray",
    normalize: bool = False,
    axis_on_off: str = "off",
    enhance_contrast: bool = False,
    gamma: float = 1.0,
    bg_threshold: float = 0.0,
 ):
    """Muestra una cuadrícula de imágenes.
    
    Args:
        images: Tensor (N, C, H, W) o lista de tensores (C, H, W).
        figsize: Tamaño de la figura (ancho, alto).
        n_rows: Número de filas en la cuadrícula.
        n_cols: Número de columnas en la cuadrícula.
        padding: Píxeles de separación entre imágenes.
        pad_value: Valor de relleno (0 = negro, 1 = blanco).
        cmap: Mapa de colores para imágenes de un solo canal.
        normalize: Si True, make_grid normaliza los valores.
        axis_on_off: "off" para ocultar ejes, "on" para mostrarlos.
        enhance_contrast: Si True, aplica normalización min-max por imagen
                          para mejorar el contraste de imágenes generadas.
        gamma: Exponente de corrección gamma (solo si enhance_contrast=True).
               gamma < 1 aclara los tonos medios, gamma > 1 los oscurece.
        bg_threshold: Umbral para suprimir el fondo (solo si enhance_contrast=True).
                      0.0 = sin supresión, 0.05-0.15 = limpieza suave del fondo.
    """

    if isinstance(images, list):
        images = torch.stack([torch.clamp(img, 0.0, 1.0) for img in images])
    else:
        images = torch.clamp(images, 0.0, 1.0)
    
    # Mejora de contraste: normalización min-max + gamma por imagen
    if enhance_contrast:
        images = _enhance_contrast(images, gamma=gamma, bg_threshold=bg_threshold)

    grid = make_grid(
        images, 
        nrow=n_cols, 
        padding=padding, 
        normalize=normalize,
        pad_value=pad_value,
    ) 

    # Convert to PIL Image and display
    fig, ax = plt.subplots(figsize=figsize)
    # If images have 3 channels, cmap is ignored by imshow
    pil_image = functional.to_pil_image(grid)
    ax.imshow(pil_image, cmap=cmap if grid.shape[0] == 1 else None)
    ax.axis(axis_on_off)
    return fig, ax


def plot_image_evolution(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps: ArrayLike,
    figsize: tuple,
    cmap: Colormap = "gray",
):
    # Usamos clamp(0, 1) para todos los frames.
    # - Frames intermedios VE (sigma≈50): valores negativos → 0 (negro),
    #   valores positivos se preservan → contraste visual real (ruido disperso).
    # - Frame final: ya está en [0,1] gracias a la normalización de pf_ode,
    #   el clamp no lo modifica.
    # La alternativa min-max produce gris uniforme en frames ruidosos (la media
    # de N(0,50) mapea a 0.5), lo que hace las imágenes "tenues".
    images_display = torch.clamp(images, 0.0, 1.0)

    fig, axs = plt.subplots(
        n_images,
        len(n_intermediate_steps),
        figsize=figsize,
    )

    for n_image in np.arange(n_images):
        for i, ax in enumerate(axs[n_image, :]):
            img = images_display[n_image, :, :, :, n_intermediate_steps[i]].cpu().numpy()
            if img.shape[0] == 1:  # Grayscale
                ax.imshow(img[0, :, :], cmap="gray", vmin=0.0, vmax=1.0)
            else:  # RGB
                # Reorder from (C, H, W) to (H, W, C)
                ax.imshow(np.transpose(img, (1, 2, 0)), vmin=0.0, vmax=1.0)
            axs[n_image, i].set_axis_off()
    return fig, axs

def animation_images(
        images_t, 
        interval,
        figsize,
    ): 
    images_t = torch.clamp(torch.tensor(images_t), 0.0, 1.0).numpy()
    
    is_color = len(images_t.shape) == 4
    if is_color:
        channels, h, w, n_frames = images_t.shape
        # Move channel to last dimension for pyplot (H, W, C, n_frames)
        images_t = np.transpose(images_t, (1, 2, 0, 3))
    else:
        h, w, n_frames = images_t.shape

    # Create a figure and axes.  
    fig, ax = plt.subplots(figsize=figsize)
    
    if is_color:
        img_display = ax.imshow(images_t[..., 0])
    else:
        img_display = ax.imshow(images_t[..., 0], cmap="gray")

    def update(t):
        """Update function for the animation."""
        img_display.set_array(images_t[..., t])
        return [img_display]

    return ( 
        fig, 
        ax, 
        animation.FuncAnimation(
            fig, 
            update, 
            frames=n_frames, 
            interval=interval, 
            blit=False)
    )
