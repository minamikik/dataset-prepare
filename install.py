import git
import os
import os.path as osp


repos = [
    {
        'url': 'https://github.com/yjl450/age-estimation-ldl-pytorch',
        'name': 'age'
    },
    {
        'url': 'https://github.com/salesforce/BLIP',
        'name': 'BLIP'
    },
    {
        'url': 'https://github.com/JingyunLiang/SwinIR',
        'name': 'SwinIR'
    }    
]

def install(repos):
    if not osp.exists('src'):
        os.makedirs('src')
    if not osp.exists('models'):
        os.makedirs('models')

    for repo in repos:
        print(f'Cloning {repo["name"]} from {repo["url"]}')
        git.Repo.clone_from(
            repo['url'],
            'src'
        )
        if repo['name'] == 'age':
            os.rename('src/age-estimation-ldl-pytorch', 'src/age')


if __name__ == '__main__':
    install(repos)
