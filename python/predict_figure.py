import matplotlib.pyplot as plt 
    ...:  
    ...: cols = ['{}'.format(col) for col in ['RGB', 'Ground Truth','Predicted']] 
    ...: rows = ['{}'.format(row) for row in (filenames)] 
    ...:  
    ...: fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(6, 16)) 
    ...:  
    ...: for ax, col in zip(axes[0], cols): 
    ...:     ax.set_title(col) 
    ...:  
    ...: for ax, row in zip(axes[:,0], rows): 
    ...:     ax.set_ylabel(row, rotation=90, size='large') 
    ...:  
    ...: counter=0 
    ...: for i, row in enumerate(axes): 
    ...:     for j, cell in enumerate(row): 
    ...:         cutout = bigArray[inds[counter]].crop((0,0,1000,1000)) 
    ...:         if (counter+1) % 3 == 0: 
    ...:             color = 'binary' 
    ...:         elif counter+1 % 3 != 0: 
    ...:             color = 'binary_r' 
    ...:         else: 
    ...:             color = None 
    ...:         cell.imshow(cutout,cmap=color)  
    ...:         counter+=1 
    ...:         cell.set_xticks([]) 
    ...:         cell.set_yticks([]) 
    ...: fig.subplots_adjust(hspace=1, wspace=0)   
    ...: fig.tight_layout() 
    ...: plt.savefig('Predicted_Grid.png',bbox_inches='tight') 
    ...: plt.show()import matplotlib.pyplot as plt 
    ...:  
    ...: cols = ['{}'.format(col) for col in ['RGB', 'Ground Truth','Predicted']] 
    ...: rows = ['{}'.format(row) for row in (filenames)] 
    ...:  
    ...: fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(6, 16)) 
    ...:  
    ...: for ax, col in zip(axes[0], cols): 
    ...:     ax.set_title(col) 
    ...:  
    ...: for ax, row in zip(axes[:,0], rows): 
    ...:     ax.set_ylabel(row, rotation=90, size='large') 
    ...:  
    ...: counter=0 
    ...: for i, row in enumerate(axes): 
    ...:     for j, cell in enumerate(row): 
    ...:         cutout = bigArray[inds[counter]].crop((0,0,1000,1000)) 
    ...:         if (counter+1) % 3 == 0: 
    ...:             color = 'binary' 
    ...:         elif counter+1 % 3 != 0: 
    ...:             color = 'binary_r' 
    ...:         else: 
    ...:             color = None 
    ...:         cell.imshow(cutout,cmap=color)  
    ...:         counter+=1 
    ...:         cell.set_xticks([]) 
    ...:         cell.set_yticks([]) 
    ...: fig.subplots_adjust(hspace=1, wspace=0)   
    ...: fig.tight_layout() 
    ...: plt.savefig('Predicted_Grid.png',bbox_inches='tight') 
    ...: plt.show()import matplotlib.pyplot as plt 
    ...:  
    ...: cols = ['{}'.format(col) for col in ['RGB', 'Ground Truth','Predicted']] 
    ...: rows = ['{}'.format(row) for row in (filenames)] 
    ...:  
    ...: fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(6, 16)) 
    ...:  
    ...: for ax, col in zip(axes[0], cols): 
    ...:     ax.set_title(col) 
    ...:  
    ...: for ax, row in zip(axes[:,0], rows): 
    ...:     ax.set_ylabel(row, rotation=90, size='large') 
    ...:  
    ...: counter=0 
    ...: for i, row in enumerate(axes): 
    ...:     for j, cell in enumerate(row): 
    ...:         cutout = bigArray[inds[counter]].crop((0,0,1000,1000)) 
    ...:         if (counter+1) % 3 == 0: 
    ...:             color = 'binary' 
    ...:         elif counter+1 % 3 != 0: 
    ...:             color = 'binary_r' 
    ...:         else: 
    ...:             color = None 
    ...:         cell.imshow(cutout,cmap=color)  
    ...:         counter+=1 
    ...:         cell.set_xticks([]) 
    ...:         cell.set_yticks([]) 
    ...: fig.subplots_adjust(hspace=1, wspace=0)   
    ...: fig.tight_layout() 
    ...: plt.savefig('Predicted_Grid.png',bbox_inches='tight') 
    ...: plt.show()                                                                                                                                                                         

import matplotlib.pyplot as plt 
    ...:  
    ...: cols = ['{}'.format(col) for col in ['RGB', 'Ground Truth','Predicted']] 
    ...: rows = ['{}'.format(row) for row in (filenames)] 
    ...:  
    ...: fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(6, 16)) 
    ...:  
    ...: for ax, col in zip(axes[0], cols): 
    ...:     ax.set_title(col) 
    ...:  
    ...: for ax, row in zip(axes[:,0], rows): 
    ...:     ax.set_ylabel(row, rotation=90, size='large') 
    ...:  
    ...: counter=0 
    ...: for i, row in enumerate(axes): 
    ...:     for j, cell in enumerate(row): 
    ...:         cutout = bigArray[inds[counter]].crop((0,0,1000,1000)) 
    ...:         if (counter+1) % 3 == 0: 
    ...:             color = 'binary' 
    ...:         elif counter+1 % 3 != 0: 
    ...:             color = 'binary_r' 
    ...:         else: 
    ...:             color = None 
    ...:         cell.imshow(cutout,cmap=color)  
    ...:         counter+=1 
    ...:         cell.set_xticks([]) 
    ...:         cell.set_yticks([]) 
    ...: fig.subplots_adjust(hspace=1, wspace=0)   
    ...: fig.tight_layout() 
    ...: plt.savefig('Predicted_Grid.png',bbox_inches='tight') 
    ...: plt.show() import matplotlib.pyplot as plt 
    ...:  
    ...: cols = ['{}'.format(col) for col in ['RGB', 'Ground Truth','Predicted']] 
    ...: rows = ['{}'.format(row) for row in (filenames)] 
    ...:  
    ...: fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(6, 16)) 
    ...:  
    ...: for ax, col in zip(axes[0], cols): 
    ...:     ax.set_title(col) 
    ...:  
    ...: for ax, row in zip(axes[:,0], rows): 
    ...:     ax.set_ylabel(row, rotation=90, size='large') 
    ...:  
    ...: counter=0 
    ...: for i, row in enumerate(axes): 
    ...:     for j, cell in enumerate(row): 
    ...:         cutout = bigArray[inds[counter]].crop((0,0,1000,1000)) 
    ...:         if (counter+1) % 3 == 0: 
    ...:             color = 'binary' 
    ...:         elif counter+1 % 3 != 0: 
    ...:             color = 'binary_r' 
    ...:         else: 
    ...:             color = None 
    ...:         cell.imshow(cutout,cmap=color)  
    ...:         counter+=1 
    ...:         cell.set_xticks([]) 
    ...:         cell.set_yticks([]) 


    ...: fig.subplots_adjust(hspace=1, wspace=0)   
    ...: fig.tight_layout() 
    ...: plt.savefig('Predicted_Grid.png',bbox_inches='tight') 
    ...: plt.show() 
